from __future__ import print_function

import numpy as np
import tensorflow as tf
import time

import relaax.algorithm_base.agent_base
import relaax.common.protocol.socket_protocol

from . import network


class Agent(relaax.algorithm_base.agent_base.AgentBase):
    def __init__(self, config, parameter_server):
        self._config = config
        self._parameter_server = parameter_server
        self._local_network = network.make(config)

        self.global_t = 0           # counter for global steps between all agents
        self.local_t = 0            # steps count for current agent worker
        self.episode_reward = 0     # score accumulator for current episode (game)

        self.states = []            # auxiliary states accumulator through batch_size = 0..N
        self.actions = []           # auxiliary actions accumulator through batch_size = 0..N
        self.rewards = []           # auxiliary rewards accumulator through batch_size = 0..N

        self.episode_t = 0          # episode counter through batch_size = 0..M
        self.latency = 0            # latency accumulator for one episode loop

        if config.preprocess:
            if type(config.state_size) not in [list, tuple]:
                self._config.state_size = [config.state_size]
            self.prev_state = np.zeros(self._config.state_size)

        self._session = tf.Session()

        self._session.run(tf.variables_initializer(tf.global_variables()))

    def act(self, state):
        start = time.time()
        if self._config.preprocess:
            state = self._update_state(state)

        if state.ndim > 1:  # lambda layer
            state = state.flatten()

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
        probs = self._local_network.run_policy(self._session, state)
        action = self.choose_action(probs)

        self.states.append(state)

        action_vec = np.zeros([self._config.action_size])  # one-hot vector to store taken action
        action_vec[action] = 1
        self.actions.append(action_vec)

        if (self.local_t % 100) == 0:   # can add by config
            print("TIMESTEP {}\nProbs: {}".format(self.local_t, probs))
            self.metrics().scalar('server latency', self.latency / 100)
            self.latency = 0

        self.latency += time.time() - start
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

        if self._config.preprocess:
            self.prev_state.fill(0)

        return score

    def _reward(self, reward):
        self.episode_reward += reward
        self.rewards.append(reward)

        self.local_t += 1
        self.episode_t += 1
        self.global_t = self._parameter_server.increment_global_t()

        return self.global_t < self._config.max_global_step

    def _update_global(self):
        feed_dict = {
            self._local_network.s: self.states,
            self._local_network.a: self.actions,
            self._local_network.advantage: self.discounted_reward(np.vstack(self.rewards)),
        }

        self._parameter_server.apply_gradients(
            self._session.run(self._local_network.grads, feed_dict=feed_dict)
        )

    @staticmethod
    def choose_action(pi_values):
        values = np.cumsum(pi_values)
        total = values[-1]
        r = np.random.rand() * total
        return np.searchsorted(values, r)

    def discounted_reward(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, r.size)):
            running_add = running_add * self._config.GAMMA + r[t]
            discounted_r[t] = running_add
        # size the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_r = discounted_r.astype(np.float64)
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r) + 1e-20
        return discounted_r

    def metrics(self):
        return self._parameter_server.metrics()

    def _update_state(self, state):
        # Computes difference from the previous observation (motion-like process)
        self.prev_state = state - self.prev_state
        return self.prev_state
