import tensorflow as tf


# Simple 2-layer fully-connected Policy Neural Network
class _PolicyNN(object):

    def __init__(self):
        self.global_t = tf.Variable(0, tf.int64)
        self.increment_global_t = tf.assign_add(self.global_t, 1)

    def compute_loss(self, config):
        return self

    def compute_gradients(self, config):
        return self

    def apply_gradients(self, config):
        return self


class _PolicyNNShared(_PolicyNN):

    def __init__(self, config):
        super(_PolicyNNShared, self).__init__()
