import math
import time
import random
import tensorflow as tf
import numpy as np
import rmsprop_applier
import accum_trainer
import game_ac_network


class Factory(object):
    def __init__(
        self,
        params,
        global_network,
        local_device,
        get_session,
        add_summary
    ):
        learning_rate_input = tf.placeholder("float")

        apply_gradients = rmsprop_applier.RMSPropApplier(
            learning_rate=learning_rate_input,
            decay=params.RMSP_ALPHA,
            momentum=0.0,
            epsilon=params.RMSP_EPSILON,
            clip_norm=params.GRAD_NORM_CLIP,
            global_vars=global_network.get_vars(),
            device=local_device
        )

        episode_score = tf.placeholder(tf.int32)
        summary = tf.scalar_summary('episode score', episode_score)

        self._n_workers = 0
        self._factory = lambda ident: _Worker(
            ident=ident,
            params=params,
            global_network=global_network,
            device=local_device,
            learning_rate_input=learning_rate_input,
            apply_gradients=apply_gradients,
            get_session=get_session,
            log_reward=lambda reward, step: add_summary(
                get_session().run(summary, feed_dict={episode_score: reward}),
                step
            )
        )

    def __call__(self):
        worker = self._factory(self._n_workers)
        self._n_workers += 1
        return worker


class _Worker(object):
    def __init__(
        self,
        ident,
        params,
        global_network,
        device,
        learning_rate_input,
        apply_gradients,
        get_session,
        log_reward
    ):

        self._ident = ident
        self._params = params
        self._global_network = global_network 
        self._get_session = get_session
        self._log_reward = log_reward
        self._learning_rate_input = learning_rate_input
        self._last_time = 0

        with tf.device(device):
            self._local_network = game_ac_network \
                .make_full_network(params, ident) \
                .prepare_loss(params)

        # TODO: don't need accum trainer anymore with batch
        trainer = accum_trainer.AccumTrainer(device)
        trainer.prepare_minimize(
            self._local_network.total_loss,
            self._local_network.get_vars()
        )

        self.accum_gradients = trainer.accumulate_gradients()
        self.reset_gradients = trainer.reset_gradients()

        self.apply_gradients = apply_gradients(trainer.get_accum_grad_list())

        self.sync = game_ac_network.assign_vars(self._local_network, global_network)

        self._initial_learning_rate = _log_uniform(
            params.INITIAL_ALPHA_LOW,
            params.INITIAL_ALPHA_HIGH,
            params.INITIAL_ALPHA_LOG_RATE
        )


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

    def act(self, state):
        sess = self._get_session()

        self.update_state(state)

        if self.episode_t == self._params.episode_len:
            self._update_global()

            if self.terminal_end:
                self.terminal_end = False

            self.episode_t = 0

        if self.episode_t == 0:
            # reset accumulated gradients
            sess.run(self.reset_gradients)
            # copy weights from shared to local
            sess.run(self.sync)

            self.states = []
            self.actions = []
            self.rewards = []
            self.values = []

            if self._params.use_LSTM:
                self.start_lstm_state = self._local_network.lstm_state_out

        pi_, value_ = self._local_network.run_policy_and_value(sess, self.frameQueue)
        action = self.choose_action(pi_)

        self.states.append(self.frameQueue)
        self.actions.append(action)
        self.values.append(value_)

        if (self._ident == 0) and (self.local_t % 100) == 0:
            print("pi=", pi_)
            print(" V=", value_)

        return action

    def on_episode(self, reward, terminal):
        sess = self._get_session()

        score = 0

        self.episode_reward += reward

        # clip reward
        self.rewards.append(np.clip(reward, -1, 1))

        self.local_t += 1
        self.episode_t += 1
        global_t = sess.run(self._global_network.increment_global_t)

        if global_t > self._params.max_global_step:
            return 0, True

        if terminal:
            self.terminal_end = True
            print("score=", self.episode_reward)

            score = self.episode_reward

            self._log_reward(self.episode_reward, global_t)

            self.episode_reward = 0

            if self._params.use_LSTM:
                self._local_network.reset_state()

            self.episode_t = self._params.episode_len

        return score, False

    def anneal_learning_rate(self, global_time_step):
        learning_rate = self._initial_learning_rate * \
                        (self._params.max_global_step - global_time_step) / self._params.max_global_step 
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

    def _update_global(self):
        sess = self._get_session()

        R = 0.0
        if not self.terminal_end:
            R = self._local_network.run_value(sess, self.frameQueue)

        self.actions.reverse()
        self.states.reverse()
        self.rewards.reverse()
        self.values.reverse()

        batch_si = []
        batch_a = []
        batch_td = []
        batch_R = []

        # compute and accumulate gradients
        for (ai, ri, si, Vi) in zip(self.actions,
                                    self.rewards,
                                    self.states,
                                    self.values):
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

            sess.run(
                self.accum_gradients,
                feed_dict={
                    self._local_network.s: batch_si,
                    self._local_network.a: batch_a,
                    self._local_network.td: batch_td,
                    self._local_network.r: batch_R,
                    self._local_network.initial_lstm_state: self.start_lstm_state,
                    self._local_network.step_size: [len(batch_a)]
                }
            )
        else:
            sess.run(
                self.accum_gradients,
                feed_dict={
                    self._local_network.s: batch_si,
                    self._local_network.a: batch_a,
                    self._local_network.td: batch_td,
                    self._local_network.r: batch_R
                }
            )

        start_time = time.time()
        cur_learning_rate = self.anneal_learning_rate(
            sess.run(self._global_network.global_t)
        )

        sess.run(
            self.apply_gradients,
            feed_dict={self._learning_rate_input: cur_learning_rate}
        )
        elapsed = time.time() - start_time
        interval = time.time() - self._last_time
        self._last_time = time.time()

        if (self._ident == 0) and (self.local_t % 100) == 0:
            print("TIMESTEP", self.local_t)
            print("ELAPSED", elapsed, interval, elapsed / interval)


def _log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)

