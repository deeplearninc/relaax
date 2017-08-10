from builtins import object
import numpy as np

from .. import dqn_config

import tensorflow as tf
from relaax.common.algorithms import subgraph

from collections import deque
import random


class ReplayBuffer(object):
    def __init__(self, max_len):
        self._replay_memory = deque(maxlen=max_len)

    def sample(self, size):
        return random.sample(self._replay_memory, size)

    def append(self, value):
        self._replay_memory.append(value)


class Action(subgraph.Subgraph):
    def build_graph(self):
        self.ph_local_step = tf.placeholder(tf.int64, [])
        self.ph_q_value = tf.placeholder(tf.float32, [None, dqn_config.config.output.action_size])

        eps = tf.train.polynomial_decay(dqn_config.config.initial_eps, self.ph_local_step, dqn_config.config.decay_steps, dqn_config.config.end_eps)
        return tf.cond(tf.less(tf.random_uniform([]), eps),
                       lambda: tf.random_uniform([], 0, dqn_config.config.output.action_size, dtype=tf.int32),
                       lambda: tf.cast(tf.squeeze(tf.argmax(self.ph_q_value, axis=1)), tf.int32))


class DQNObservation(object):
    def __init__(self):
        self.queue = None

    def add_state(self, state):
        if state is None:
            self.queue = None
        else:
            self.queue = np.array(state)
            return

            state = np.array(state)
            axis = len(state.shape)  # extra dimension for observation

            # observation = np.reshape(state, state.shape + (1,))
            observation = np.expand_dims(state, axis)

            if self.queue is None:
                self.queue = np.repeat(observation, dqn_config.config.input.history, axis=axis)
            else:
                # remove oldest observation from the beginning of the observation queue
                self.queue = np.delete(self.queue, 0, axis=axis)

                # append latest observation to the end of the observation queue
                self.queue = np.append(self.queue, observation, axis=axis)
