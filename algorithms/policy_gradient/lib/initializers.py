import tensorflow as tf
import numpy as np


class Zero(object):
    def __call__(self, shape=None, dtype=np.float):
        return np.zeros(shape=shape, dtype=dtype)


class Xavier(object):
    DTYPES = {
        np.float: tf.float64,
        np.float64: tf.float64,
        np.float32: tf.float32
    }

    def __call__(self, shape, dtype=np.float):
        return tf.contrib.layers.xavier_initializer()(
            shape=shape,
            dtype=self.DTYPES[dtype]
        )
