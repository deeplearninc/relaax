import tensorflow as tf


class Subgraph(object):
    def __init__(self, *args, **kwargs):
        with tf.variable_scope(type(self).__name__):
            self.__pointer = self.build(*args, **kwargs)

    @property
    def tensor(self):
        return self.__pointer

    @property
    def op(self):
        return self.__pointer
