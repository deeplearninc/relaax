import numpy as np
import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph


class ValidBorder(object):
    PADDING = 'VALID'


class SameBorder(object):
    PADDING = 'SAME'


class Flatten(subgraph.Subgraph):
    def build_graph(self, x):
        return graph.Reshape(x, (-1, np.prod(x.node.shape.as_list()[1:]))).node


class Convolution(subgraph.Subgraph):
    def build_graph(self, x, n_filters, filter_size, stride,
            border=ValidBorder, activation=graph.Relu):

        self.weight = graph.Wb(np.float32,
                filter_size + [x.node.shape.as_list()[-1], n_filters])

        return activation(graph.TfNode(tf.nn.conv2d(x.node, self.weight.W,
                strides=[1] + stride + [1], padding=border.PADDING) +
                self.weight.b)).node


class Dense(subgraph.Subgraph):
    def build_graph(self, x, size, activation=graph.Relu):
        assert len(x.node.shape) == 2
        self.weight = graph.Wb(np.float32, (x.node.shape.as_list()[1], size))
        return activation(graph.ApplyWb(x, self.weight)).node
