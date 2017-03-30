import tensorflow as tf


class FullyConnected(object):
    """Builds fully connected neural network."""

    @classmethod
    def assemble(cls, t_input, t_output, hidden_layers_size):
        pass

    @classmethod
    def assemble_from_weights(cls, t_input, t_weights):
        layer = tf.nn.relu(tf.matmul(t_input, t_weights[0]))
        for i in range(1, len(t_weights) - 1):
            layer = tf.nn.relu(tf.matmul(layer, t_weights[i]))
        return tf.nn.softmax(tf.matmul(layer, t_weights[-1]))
