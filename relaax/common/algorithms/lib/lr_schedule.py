from __future__ import division
import numpy as np
import tensorflow as tf

from relaax.common.algorithms import subgraph


class Linear(subgraph.Subgraph):
    def build_graph(self, global_step, cfg):
        n_steps = np.int64(cfg.max_global_step)
        reminder = tf.subtract(n_steps, global_step.n.node)
        factor = tf.cast(reminder, tf.float64) / tf.cast(n_steps, tf.float64)

        assert cfg.learning_rate_end < cfg.initial_learning_rate, \
            "learning_rate_end should be lower than initial_learning_rate"

        learning_rate = \
            cfg.initial_learning_rate - factor * (cfg.initial_learning_rate - cfg.learning_rate_end)
        learning_rate = tf.maximum(tf.cast(cfg.learning_rate_end, tf.float64), learning_rate)

        return learning_rate
