import tensorflow as tf


class FullyConnected(object):
    """Builds fully connected neural network."""

    @classmethod
    def assemble(cls, state, weights):
        last = state
        for w in weights:
            last = tf.nn.relu(tf.matmul(last, w))
        return tf.nn.softmax(last)
