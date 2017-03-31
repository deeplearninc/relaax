import tensorflow as tf

from relaax.common.algorithms.subgraph import Subgraph


class ApplyGradients(Subgraph):
    def build(self, optimizer, gradients, weights):
        return optimizer.node.apply_gradients(
            zip(gradients.node, weights.node)
        )


class Adam(Subgraph):
    def build(self, learning_rate=0.001):
        return tf.train.AdamOptimizer(learning_rate=learning_rate)
