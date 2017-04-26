import tensorflow as tf


class Subgraph(object):

    def __init__(self, *args, **kwargs):
        with tf.variable_scope(type(self).__name__):
            self.__node = self.build_graph(*args, **kwargs)

    @property
    def node(self):
        return self.__node

    class Op(object):
        def __init__(self, subgraph, **kwargs):
            self.subgraph = subgraph
            self.feed_dict = kwargs
