import tensorflow as tf
import random
from accum_trainer import AccumTrainer
import game_ac_network
import numpy as np


class Worker(object):
    def __init__(self, params,
                 thread_index,
                 global_network,
                 initial_learning_rate,
                 learning_rate_input,
                 grad_applier,
                 max_global_time_step,
                 device):

        self._params = params
        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step

        with tf.device(device):
            self.local_network = game_ac_network \
                .make_full_network(params, thread_index) \
                .prepare_loss(params)

        # TODO: don't need accum trainer anymore with batch
        self.trainer = AccumTrainer(device)
        self.trainer.prepare_minimize(
            self.local_network.total_loss,
            self.local_network.get_vars()
        )

        self.accum_gradients = self.trainer.accumulate_gradients()
        self.reset_gradients = self.trainer.reset_gradients()

        self.apply_gradients = grad_applier(
            global_network.get_vars(),
            self.trainer.get_accum_grad_list()
        )

        self.sync = game_ac_network.assign_vars(self.local_network, global_network)

        self._initial_learning_rate = initial_learning_rate

        self.local_t = 0            # steps count for current agent's thread
        self.episode_reward = 0     # score accumulator for current game

        self.states = []            # auxiliary states accumulator through episode_len = 0..5
        self.actions = []           # auxiliary actions accumulator through episode_len = 0..5
        self.rewards = []           # auxiliary rewards accumulator through episode_len = 0..5
        self.values = []            # auxiliary values accumulator through episode_len = 0..5
        self.start_lstm_state = None
        self.episode_t = 0          # episode counter through episode_len = 0..5
        self.terminal_end = False   # auxiliary parameter to compute R in update_global and frameQueue

        self.frameQueue = None      # frame accumulator for state, cuz state = 4 consecutive frames

    def act(self, trainer, state):
        self.update_state(state)

        if self.episode_t == self._params.episode_len:
            self._update_global(trainer)

            if self.terminal_end:
                self.terminal_end = False

            self.episode_t = 0

        if self.episode_t == 0:
            trainer.sess.run(self.local_network.copy_W_fc3)

            # reset accumulated gradients
            trainer.sess.run(self.reset_gradients)
            # nprint('accum_grad_list zeros', trainer.sess.run(self.trainer.get_accum_grad_list()))
            # copy weights from shared to local
            trainer.sess.run(self.sync)

            self.states = []
            self.actions = []
            self.rewards = []
            self.values = []

            if self._params.use_LSTM:
                self.start_lstm_state = self.local_network.lstm_state_out

        pi_, value_ = self.local_network.run_policy_and_value(trainer.sess, self.frameQueue)
        action = self.choose_action(pi_)

        self.states.append(self.frameQueue)
        self.actions.append(action)
        self.values.append(value_)

        if (self.thread_index == 0) and (self.local_t % 100) == 0:
            print("pi=", pi_)
            print(" V=", value_)

        return action

    def on_episode(self, trainer, reward, terminal):
        score = 0

        self.episode_reward += reward

        # clip reward
        self.rewards.append(np.clip(reward, -1, 1))

        self.local_t += 1
        self.episode_t += 1
        global_t = trainer.sess.run(trainer.global_network.increment_global_t)

        if global_t > self._params.max_global_step:
            return 0, True

        if terminal:
            self.terminal_end = True
            print("score=", self.episode_reward)

            score = self.episode_reward

            trainer._summary_writer.add_summary(
                trainer.sess.run(trainer._summary, feed_dict={
                    trainer._episode_score: self.episode_reward
                }),
                global_t
            )

            self.episode_reward = 0

            if self._params.use_LSTM:
                self.local_network.reset_state()

            self.episode_t = self._params.episode_len

        return score, False

    def anneal_learning_rate(self, global_time_step):
        learning_rate = self._initial_learning_rate * \
                        (self.max_global_time_step - global_time_step) / self.max_global_time_step
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate

    @staticmethod
    def choose_action(pi_values):
        values = []
        total = 0.0
        for rate in pi_values:
            total += rate
            value = total
            values.append(value)

        r = random.random() * total
        for i in range(len(values)):
            if values[i] >= r:
                return i
        # fail safe
        return len(values) - 1

    def update_state(self, frame):
        if not self.terminal_end and self.local_t != 0:
            self.frameQueue = np.append(self.frameQueue[:, :, 1:], frame, axis=2)
        else:
            self.frameQueue = np.stack((frame, frame, frame, frame), axis=2)

    @staticmethod   # NOT USED YET --> need for TensorBoard
    def _record_score(sess, summary_writer, summary_op, score_input, score, global_t):
        summary_str = sess.run(summary_op, feed_dict={
            score_input: score
        })
        summary_writer.add_summary(summary_str, global_t)

    def _update_global(self, trainer):
        worker = self
        R = 0.0
        if not self.terminal_end:
            R = worker.local_network.run_value(trainer.sess, self.frameQueue)

        worker.actions.reverse()
        worker.states.reverse()
        worker.rewards.reverse()
        worker.values.reverse()

        batch_si = []
        batch_a = []
        batch_td = []
        batch_R = []

        # compute and accumulate gradients
        for (ai, ri, si, Vi) in zip(worker.actions,
                                    worker.rewards,
                                    worker.states,
                                    worker.values):
            R = ri + self._params.GAMMA * R
            td = R - Vi
            a = np.zeros([self._params.action_size])
            a[ai] = 1

            batch_si.append(si)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)

        if self._params.use_LSTM:
            batch_si.reverse()
            batch_a.reverse()
            batch_td.reverse()
            batch_R.reverse()

            trainer.sess.run(
                worker.accum_gradients,
                feed_dict={
                    worker.local_network.s: batch_si,
                    worker.local_network.a: batch_a,
                    worker.local_network.td: batch_td,
                    worker.local_network.r: batch_R,
                    worker.local_network.initial_lstm_state: worker.start_lstm_state,
                    worker.local_network.step_size: [len(batch_a)]
                }
            )
        else:
            trainer.sess.run(
                worker.accum_gradients,
                feed_dict={
                    worker.local_network.s: batch_si,
                    worker.local_network.a: batch_a,
                    worker.local_network.td: batch_td,
                    worker.local_network.r: batch_R
                }
            )

        cur_learning_rate = worker.anneal_learning_rate(
            trainer.sess.run(trainer.global_network.global_t)
        )

        trainer.sess.run(trainer.global_network.copy_W_fc3)
        trainer.sess.run(
            worker.apply_gradients,
            feed_dict={worker.learning_rate_input: cur_learning_rate}
        )

        if (self.thread_index == 0) and (worker.local_t % 100) == 0:
            print("TIMESTEP", worker.local_t)

