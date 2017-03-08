from __future__ import print_function

import tensorflow as tf
import time
from collections import defaultdict

import keras.backend

import relaax.algorithm_base.agent_base
import relaax.common.protocol.socket_protocol

from . import network


class Agent(relaax.algorithm_base.agent_base.AgentBase):
    def __init__(self, config, parameter_server):
        self._config = config
        self._parameter_server = parameter_server

        self._episode_timestep = 0   # timestep for current episode (round)
        self._episode_reward = 0     # score accumulator for current episode (round)
        self._stop_training = False  # stop training flag to prevent the training further

        self.data = defaultdict(list)

        # inform Keras that we are going to initialize variables here
        keras.backend.manual_variable_initialization(True)

        self._session = tf.Session()
        keras.backend.set_session(self._session)

        self.policy_net, value_net = network.make(config)
        self.policy, _ = network.make_head(config, self.policy_net, value_net, self._session)

        self._session.run(tf.variables_initializer(tf.global_variables()))

        self._n_iter = self._parameter_server.wait_for_iteration()  # counter for global updates at parameter server
        self.policy.net.set_weights(
            list(self._parameter_server.receive_weights(self._n_iter))
        )

        if config.use_filter:
            self.obs_filter, _ = network.make_filters(config)
            state = self._parameter_server.get_filter_state()
            print(state[0])
            print(state[1])
            self.obs_filter.set(self._parameter_server.get_global_t(), state[0], state[1])

        self.server_latency_accumulator = 0     # accumulator for averaging server latency
        self.collecting_time = time.time()      # timer for collecting experience

    def act(self, obs):
        start = time.time()

        if self._config.use_filter:
            obs = self.obs_filter(obs)
        self.data["observation"].append(obs)

        action, agentinfo = self.policy.act(obs)
        self.data["action"].append(action)

        for (k, v) in agentinfo.iteritems():
            self.data[k].append(v)

        self.server_latency_accumulator += time.time() - start
        return action

    def reward_and_act(self, reward, state):
        if not self._reward(reward):
            return self.act(state)
        return None

    def reward_and_reset(self, reward):
        if self._reward(reward):
            return None

        score = self._episode_reward
        print("Episode reward =", score)
        self.metrics().scalar('episode reward', score)

        latency = self.server_latency_accumulator / self._episode_timestep
        self.server_latency_accumulator = 0
        self.metrics().scalar('server latency', latency)

        self._send_experience(terminated=(self._episode_timestep < self._config.timestep_limit))
        return score

    def _reward(self, reward):
        self._episode_reward += reward

        # reward = self.reward_filter(reward)
        self.data["reward"].append(reward)

        self._episode_timestep += 1

        return self._stop_training

    def _send_experience(self, terminated=False):
        self.data["terminated"] = terminated
        self._parameter_server.send_experience(self._n_iter, self.data, self._episode_timestep)

        self.data.clear()
        self._episode_timestep = 0
        self._episode_reward = 0

        old_n_iter = self._n_iter
        self._n_iter = self._parameter_server.wait_for_iteration()
        if self._n_iter == -1:
            self._n_iter = old_n_iter
            return

        if self._n_iter > self._config.n_iter:
            self._stop_training = True
            return

        if old_n_iter < self._n_iter:
            print('Collecting time for {} iteration: {}'.format(old_n_iter+1, time.time() - self.collecting_time))
            self.policy.net.set_weights(list(self._parameter_server.receive_weights(self._n_iter)))
            self.collecting_time = time.time()

        if self._config.use_filter:
            state = self._parameter_server.get_filter_state()
            print(state[0])
            print(state[1])
            self.obs_filter.set(self._parameter_server.get_global_t(), state[0], state[1])

    def metrics(self):
        return self._parameter_server.metrics()
