import tensorflow as tf
import random
from accum_trainer import AccumTrainer
import game_ac_network
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
import numpy as np


class A3CTrainingThread(object):
    def __init__(self, params,
                 thread_index,
                 global_network,
                 initial_learning_rate,
                 learning_rate_input,
                 grad_applier,
                 max_global_time_step,
                 device):

        self.thread_index = thread_index
        self.learning_rate_input = learning_rate_input
        self.max_global_time_step = max_global_time_step

        with tf.device(device):
            self.local_network = game_ac_network \
                .make_full_network(params, thread_index) \
                .prepare_loss(params)

        # TODO: don't need accum trainer anymore with batch
        trainer = AccumTrainer(device)
        trainer.prepare_minimize(
            self.local_network.total_loss,
            self.local_network.get_vars()
        )

        self.accum_gradients = trainer.accumulate_gradients()
        self.reset_gradients = trainer.reset_gradients()

        self.apply_gradients = grad_applier(
            global_network.get_vars(),
            trainer.get_accum_grad_list()
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
