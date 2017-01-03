from __future__ import print_function

import time
import numpy as np
import tensorflow as tf

from . import network
from .stats import ZFilter


class Agent(object):
    def __init__(self, config, parameter_server, log_dir):
        self._config = config
        self._parameter_server = parameter_server
        self._log_dir = log_dir

        kernel = "/cpu:0"
        if config.use_GPU:
            kernel = "/gpu:0"

        with tf.device(kernel):
            self._local_network = network.make(config)

        self.global_t = 0           # counter for global steps between all agents
        self.local_t = 0            # steps count for current agent's process
        self.episode_reward = 0     # score accumulator for current episode
        self.act_latency = .0       # latency summarizer

        self.states = []            # auxiliary states accumulator through episode_len = 0..5
        self.actions = []           # auxiliary actions accumulator through episode_len = 0..5
        self.rewards = []           # auxiliary rewards accumulator through episode_len = 0..5
        self.values = []            # auxiliary values accumulator through episode_len = 0..5

        self.episode_t = 0          # episode counter through episode_len = 0..5
        self.terminal_end = False   # auxiliary parameter to compute R in update_global and obsQueue
        self.start_lstm_state = None

        self.obsQueue = None        # observation accumulator, cuz state = history_len * consecutive observations
        self.obfilter = ZFilter((24,), clip=5)  # TODO: move 24 to yaml (may be a clip value)

        episode_score = tf.placeholder(tf.int32)
        summary = tf.scalar_summary('episode score', episode_score)

        act_latency = tf.placeholder(tf.float32)
        act_summary = tf.scalar_summary('act_latency', act_latency)

        initialize_all_variables = tf.initialize_all_variables()

        self._session = tf.Session()

        summary_writer = tf.train.SummaryWriter(self._log_dir, self._session.graph)

        self._session.run(initialize_all_variables)

        self._log_reward = lambda: summary_writer.add_summary(
            self._session.run(summary, feed_dict={episode_score: self.episode_reward}),
            self.global_t
        )
        self._log_latency = lambda: summary_writer.add_summary(
            self._session.run(act_summary, feed_dict={act_latency: self.act_latency}),
            self.global_t
        )

    def act(self, state):
        self.act_latency = time.time()
        self.update_state(state)

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

        mu_, sig_, value_ = self._local_network.run_policy_and_value(self._session, self.obsQueue)
        action = self._choose_action(mu_, sig_)

        self.states.append(self.obsQueue)
        self.actions.append(action)
        self.values.append(value_)

        if (self.local_t % 100) == 0:
            print("mu=", mu_)
            print("sigma=", sig_)
            print(" V=", value_)

        self.act_latency = time.time() - self.act_latency
        self._log_latency()

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

    def _choose_action(self, mu, sig):
        return (np.random.randn(1, self._config.action_size).astype(np.float32) * sig + mu)[0]

    def update_state(self, observation):
        # TODO: add history len for stacking
        self.obsQueue = self.obfilter(observation)

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

            batch_si.append(si)
            batch_a.append(ai)
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
