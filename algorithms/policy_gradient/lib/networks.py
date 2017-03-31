import tensorflow as tf

from relaax.common.algorithms.subgraph import Subgraph


class FullyConnected(Subgraph):
    """Builds fully connected neural network."""

    def build(self, state, weights):
        last = state.op
        for w in weights.op:
            last = tf.nn.relu(tf.matmul(last, w))
        return tf.nn.softmax(last)
