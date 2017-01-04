from __future__ import print_function

import numpy as np
import random
import tensorflow as tf

from . import network


class Agent(object):
    def __init__(self, config, parameter_server, log_dir):
        self._config = config
        self._parameter_server = parameter_server
        self._log_dir = log_dir

        kernel = "/cpu:0"
        if config.use_GPU:
            kernel = "/gpu:0"

        with tf.device(kernel):
            self._local_network = network.make(config, 0)

        self.local_t = 0            # steps count for current agent's thread
        self.episode_reward = 0     # score accumulator for current game

        self.states = []            # auxiliary states accumulator through episode_len = 0..5
        self.actions = []           # auxiliary actions accumulator through episode_len = 0..5
        self.rewards = []           # auxiliary rewards accumulator through episode_len = 0..5
        self.values = []            # auxiliary values accumulator through episode_len = 0..5
        self.start_lstm_state = None
        self.episode_t = 0          # episode counter through episode_len = 0..5
        self.terminal_end = False   # auxiliary parameter to compute R in update_global and frameQueue

        obs_dims = (84, 84)
        state_dims = obs_dims + (config.history_len,)
        self.dims = len(state_dims)

        self.obsQueue = None      # observation accumulator for state = history_len * consecutive frames

        episode_score = tf.placeholder(tf.int32)
        summary = tf.scalar_summary('episode score', episode_score)

        initialize_all_variables = tf.initialize_all_variables()

        self._session = tf.Session()

        summary_writer = tf.train.SummaryWriter(self._log_dir, self._session.graph)

        self._session.run(initialize_all_variables)

        self._log_reward = lambda: summary_writer.add_summary(
            self._session.run(summary, feed_dict={episode_score: self.episode_reward}),
            self.global_t
        )

    def act(self, state):
        self._update_state(state)

        if self.episode_t == self._config.episode_len:
            self._update_global()

            if self.terminal_end:
                self.terminal_end = False

            self.episode_t = 0

        if self.episode_t == 0:
            # copy weights from shared to local
            self._local_network.assign_values(self._session, self._parameter_server.get_values())

            self.states = []
            self.actions = []
            self.rewards = []
            self.values = []

            if self._config.use_LSTM:
                self.start_lstm_state = self._local_network.lstm_state_out

        pi_, value_ = self._local_network.run_policy_and_value(self._session, self.obsQueue)
        action = self._choose_action(pi_)

        self.states.append(self.obsQueue)
        self.actions.append(action)
        self.values.append(value_)

        if (self.local_t % 100) == 0:
            print("pi=", pi_)
            print(" V=", value_)

        return action

    def reward_and_act(self, reward, state):
        if self._reward(reward):
            return self.act(state)
        return None

    def reward_and_reset(self, reward):
        if not self._reward(reward):
            return None

        self.terminal_end = True
        print("score=", self.episode_reward)

        score = self.episode_reward

        self._log_reward()

        self.episode_reward = 0

        if self._config.use_LSTM:
            self._local_network.reset_state()

        self.episode_t = self._config.episode_len

        return score

    def _reward(self, reward):
        self.episode_reward += reward

        # clip reward
        self.rewards.append(np.clip(reward, -1, 1))

        self.local_t += 1
        self.episode_t += 1
        self.global_t = self._parameter_server.increment_global_t()

        return self.global_t < self._config.max_global_step

    @staticmethod
    def _choose_action(pi_values):
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
            self.obsQueue = np.append(
                self.obsQueue[:, :, 1:],
                np.reshape(frame, frame.shape + (1, )),
                axis=2
            )
        else:
            self.obsQueue = np.stack((frame, frame, frame, frame), axis=2)

    def _update_state(self, obs):
        if not self.terminal_end and self.local_t != 0:
            np.delete(self.obsQueue, 0, self.dims - 1)
            np.append(self.obsQueue,
                      np.reshape(obs, obs.shape + (1,)),
                      axis=self.dims - 1)
        else:
            self.obsQueue = np.repeat(np.reshape(obs, obs.shape + (1,)),
                                      self._config.history_len,
                                      axis=self.dims - 1)

    def _update_global(self):
        R = 0.0
        if not self.terminal_end:
            R = self._local_network.run_value(self._session, self.obsQueue)

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
            R = ri + self._config.GAMMA * R
            td = R - Vi
            a = np.zeros([self._config.action_size])
            a[ai] = 1

            batch_si.append(si)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)

        if self._config.use_LSTM:
            batch_si.reverse()
            batch_a.reverse()
            batch_td.reverse()
            batch_R.reverse()

            feed_dict = {
                    self._local_network.s: batch_si,
                    self._local_network.a: batch_a,
                    self._local_network.td: batch_td,
                    self._local_network.r: batch_R,
                    self._local_network.initial_lstm_state: self.start_lstm_state,
                    self._local_network.step_size: [len(batch_a)]
            }
        else:
            feed_dict = {
                    self._local_network.s: batch_si,
                    self._local_network.a: batch_a,
                    self._local_network.td: batch_td,
                    self._local_network.r: batch_R
            }


        self._parameter_server.apply_gradients(
            self._session.run(self._local_network.grads, feed_dict=feed_dict)
        )

        if (self.local_t % 100) == 0:
            print("TIMESTEP", self.local_t)
