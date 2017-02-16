from __future__ import print_function

import numpy as np
import tensorflow as tf
import time

import relaax.algorithm_base.agent_base
import relaax.common.protocol.socket_protocol
from network import AgentPolicyNN


def make_network(config):
    network = AgentPolicyNN(config)
    return network.prepare_loss().compute_gradients()


class Agent(relaax.algorithm_base.agent_base.AgentBase):
    def __init__(self, config, parameter_server):
        self._config = config
        self._parameter_server = parameter_server
        self._local_network = make_network(config)

        self.global_t = 0           # counter for global steps between all agents
        self.episode_reward = 0     # score accumulator for current episode (game)

        self.states = []            # auxiliary states accumulator through batch_size = 0..N
        self.actions = []           # auxiliary actions accumulator through batch_size = 0..N
        self.rewards = []           # auxiliary rewards accumulator through batch_size = 0..N

        self.episode_t = 0          # episode counter through batch_size = 0..M

        initialize_all_variables = tf.variables_initializer(tf.global_variables())

        self._session = tf.Session()

        self._session.run(initialize_all_variables)

    def act(self, state):
        start = time.time()

        if self.episode_t == self._config.batch_size:
            self._update_global()
            self.episode_t = 0

        if self.episode_t == 0:
            # copy weights from shared to local
            self._local_network.assign_values(self._session, self._parameter_server.get_values())

            self.states = []
            self.actions = []
            self.rewards = []

        # Run the policy network and get an action to take
        prob = self._local_network.run_policy(self._session, state)
        action = 0 if np.random.uniform() < prob else 1

        self.states.append(state)
        self.actions.append([action])

        self.metrics().scalar('server latency', time.time() - start)

        return action

    def reward_and_act(self, reward, state):
        if self._reward(reward):
            return self.act(state)
        return None

    def reward_and_reset(self, reward):
        if not self._reward(reward):
            return None

        print("Score =", self.episode_reward)
        score = self.episode_reward

        self.metrics().scalar('episode reward', self.episode_reward)

        self.episode_reward = 0
        self.episode_t = self._config.batch_size

        return score

    def _reward(self, reward):
        self.episode_reward += reward
        self.rewards.append(reward)

        self.episode_t += 1
        self.global_t = self._parameter_server.increment_global_t()

        return self.global_t < self._config.max_global_step

    def _update_global(self):
        feed_dict = {
            self._local_network.s: self.states,
            self._local_network.a: self.actions,
            self._local_network.advantage: self.discounted_reward(np.asarray(self.rewards)),
        }

        self._parameter_server.apply_gradients(
            self._session.run(self._local_network.grads, feed_dict=feed_dict)
        )

    def discounted_reward(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, r.size)):
            running_add = running_add * self._config.GAMMA + r[t]
            discounted_r[t] = running_add
        # size the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)
        # check zero_division, if you want
        return discounted_r

    def metrics(self):
        return self._parameter_server.metrics()
