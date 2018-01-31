from builtins import object

from collections import deque

import numpy as np
import tensorflow as tf

from .. import dqn_config
from relaax.common.algorithms import subgraph


class ReplayBuffer(object):
    def __init__(self, max_len, alpha=None):
        self._replay_memory = deque(maxlen=max_len)
        self._alpha = alpha

        self._weights = np.power(np.arange(1, self._replay_memory.maxlen + 1), self._alpha)

    def sample(self, size):
        if self._alpha is None or self._alpha == 0.0:
            return np.random.choice(self._replay_memory, size, False)
        else:
            weights = self._weights[:len(self._replay_memory)]
            return np.random.choice(self._replay_memory, size, False,
                                    p=weights / np.sum(weights))

    def append(self, value):
        self._replay_memory.append(value)


class Actor(subgraph.Subgraph):
    def build_graph(self):
        self.ph_local_step = tf.placeholder(tf.int64, [])
        self.ph_q_value = tf.placeholder(tf.float32, [None, dqn_config.config.output.action_size])

        if dqn_config.config.eps.stochastic:
            decay_steps = int(np.random.uniform(*dqn_config.config.eps.decay_steps))
        else:
            decay_steps = dqn_config.config.eps.decay_steps

        eps = tf.train.polynomial_decay(dqn_config.config.eps.initial,
                                        self.ph_local_step,
                                        decay_steps,
                                        dqn_config.config.eps.end)

        if dqn_config.config.output.q_values:
            return tf.cond(tf.less(tf.random_uniform([]), eps),
                           lambda: tf.random_uniform([dqn_config.config.output.action_size]),
                           lambda: tf.squeeze(self.ph_q_value))
        else:
            return tf.cond(tf.less(tf.random_uniform([]), eps),
                           lambda: tf.random_uniform([], 0, dqn_config.config.output.action_size,
                                                     dtype=tf.int32),
                           lambda: tf.cast(tf.squeeze(tf.argmax(self.ph_q_value, axis=1)), tf.int32))
