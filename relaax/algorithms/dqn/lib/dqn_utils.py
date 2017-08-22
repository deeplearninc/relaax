from builtins import object

from .. import dqn_config

import tensorflow as tf
from relaax.common.algorithms import subgraph

from collections import deque
import random
import math


class ReplayBuffer(object):
    def __init__(self, max_len, alpha=None):
        self._replay_memory = deque(maxlen=max_len)
        self._alpha = alpha

        self._weights = None

    def sample(self, size):
        if self._alpha is None or self._alpha == 0.0:
            return random.sample(self._replay_memory, size)
        else:
            weights = map(lambda x: math.pow(x, self._alpha), range(1, len(self._replay_memory) + 1)) if self._weights is None else self._weights
            return random.choices(self._replay_memory,
                                  weights=weights,
                                  k=size)

    def append(self, value):
        self._replay_memory.append(value)
        if len(self._replay_memory) == self._replay_memory.maxlen:
            self._weights = map(lambda x: math.pow(x, self._alpha), range(1, self._replay_memory.maxlen + 1))


class Actor(subgraph.Subgraph):
    def build_graph(self):
        self.ph_local_step = tf.placeholder(tf.int64, [])
        self.ph_q_value = tf.placeholder(tf.float32, [None, dqn_config.config.output.action_size])

        if dqn_config.config.eps.stochastic:
            decay_steps = int(random.uniform(*dqn_config.config.eps.decay_steps))
        else:
            decay_steps = dqn_config.config.eps.decay_steps

        eps = tf.train.polynomial_decay(dqn_config.config.eps.initial, self.ph_local_step, decay_steps, dqn_config.config.eps.end)
        return tf.cond(tf.less(tf.random_uniform([]), eps),
                       lambda: tf.random_uniform([], 0, dqn_config.config.output.action_size, dtype=tf.int32),
                       lambda: tf.cast(tf.squeeze(tf.argmax(self.ph_q_value, axis=1)), tf.int32))
