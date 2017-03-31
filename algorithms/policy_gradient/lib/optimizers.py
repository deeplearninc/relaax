import tensorflow as tf

from relaax.common.algorithms.subgraph import Subgraph


class Adam(object):
    def __init__(self, learning_rate=0.001):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    def apply_gradients(self):
        return ApplyGradients(self.optimizer)


class ApplyGradients(Subgraph):
    def __init__(self, optimizer):
        super(ApplyGradients, self).__init__()
        self.optimizer = optimizer

    def assemble(self, gradients, weights):
        return self.optimizer.apply_gradients(zip(gradients, weights))
