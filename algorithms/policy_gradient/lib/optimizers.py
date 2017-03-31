import tensorflow as tf

from relaax.common.algorithms.subgraph import Subgraph


class Adam(object):
    def __init__(self, learning_rate=0.001):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    def apply_gradients(self, gradients, weights):
        return ApplyGradients(self.optimizer, gradients, weights)


class ApplyGradients(Subgraph):
    def build(self, optimizer, gradients, weights):
        return optimizer.apply_gradients(zip(gradients.node, weights.node))
