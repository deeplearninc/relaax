from __future__ import division
import numpy as np
import tensorflow as tf

from relaax.common.algorithms import subgraph


class Linear(subgraph.Subgraph):
    def build_graph(self, global_step, initial_learning_rate, max_global_step, learning_rate_end=None):
        n_steps = np.int64(max_global_step)
        reminder = tf.subtract(n_steps, global_step.n.node)
        factor = tf.cast(reminder, tf.float64) / tf.cast(n_steps, tf.float64)

        if learning_rate_end is None:
            learning_rate_end = 0.2 * initial_learning_rate
        else:
            assert learning_rate_end < initial_learning_rate, \
                "learning_rate_end should be lower than initial_learning_rate"
        learning_rate = initial_learning_rate - factor * (initial_learning_rate - learning_rate_end)

        learning_rate = tf.maximum(tf.cast(learning_rate_end, tf.float64), learning_rate)
        return learning_rate
