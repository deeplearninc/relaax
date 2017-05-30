from __future__ import division
import numpy as np
import tensorflow as tf

from relaax.common.algorithms import subgraph

from .. import da3c_config


class LearningRate(subgraph.Subgraph):
    def build_graph(self, global_step):
        n_steps = np.int64(da3c_config.config.max_global_step)
        reminder = tf.subtract(n_steps, global_step.n.node)
        factor = tf.cast(reminder, tf.float64) / tf.cast(n_steps, tf.float64)
        learning_rate = tf.maximum(tf.cast(0, tf.float64), factor * da3c_config.config.initial_learning_rate)
        return learning_rate
