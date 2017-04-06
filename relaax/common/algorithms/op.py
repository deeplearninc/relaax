import tensorflow as tf


class Op(object):
    def __init__(self, subgraph, **kwargs):
        self.subgraph = subgraph
        self.feed_dict = kwargs
