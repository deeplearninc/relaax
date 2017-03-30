import tensorflow as tf


class Utils(object):
    @staticmethod
    def placeholder(shape):
        return tf.placeholder(shape=shape)

    @staticmethod
    def weights(shape):
        return tf.Variable(shape=shape)
