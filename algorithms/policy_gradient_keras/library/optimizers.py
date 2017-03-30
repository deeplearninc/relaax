import tensorflow as tf


class Optimizers(object):
    @staticmethod
    def adam(learning_rate):
        return Adam(learning_rate=learning_rate)


class Adam(object):
    def __init__(self, learning_rate):
        self.adam = tf.train.AdamOptimizer(learning_rate=learning_rate)

    def apply_gradients(gradients, weights):
        return self.adam.apply_gradients(zip(gradients, weights))
