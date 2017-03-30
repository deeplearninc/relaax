import tensorflow as tf


class Layers(object):
    @staticmethod
    def fully_connected(input, shape, activation=None, init=None)
        return tf.layers.dense(
            inputs=input,
            units=?,
            activation=activation,
            kernel_initializer=init
        )

