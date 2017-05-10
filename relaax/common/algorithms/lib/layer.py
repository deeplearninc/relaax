from __future__ import division
from past.utils import old_div
from builtins import object
import numpy as np
import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph


class Activation(object):
    @staticmethod
    def Null(x):
        return x

    @staticmethod
    def Relu(x):
        return tf.nn.relu(x)

    @staticmethod
    def Softmax(x):
        return tf.nn.softmax(x)


class Border(object):
    Valid = 'VALID'
    Same = 'SAME'


class BaseLayer(subgraph.Subgraph):
    def build_graph(self, x, shape, transformation, activation):
        d = old_div(1.0, np.sqrt(np.prod(shape[:-1])))
        initializer = graph.RandomUniformInitializer(minval=-d, maxval=d)
        W = graph.Variable(initializer(np.float32, shape)).node
        b = graph.Variable(initializer(np.float32, shape[-1:])).node
        self.weight = graph.TfNode((W, b))
        return activation(transformation(x.node, W) + b)


class Convolution(BaseLayer):
    def build_graph(self, x, n_filters, filter_size, stride,
            border=Border.Valid, activation=Activation.Relu):
        shape = filter_size + [x.node.shape.as_list()[-1], n_filters]
        tr = lambda x, W: tf.nn.conv2d(x, W, strides=[1] + stride + [1],
                    padding=border)
        return super(Convolution, self).build_graph(x, shape, tr, activation)


class Dense(BaseLayer):
    def build_graph(self, x, size, activation=Activation.Null):
        assert len(x.node.shape) == 2
        shape = (x.node.shape.as_list()[1], size)
        tr = lambda x, W: tf.matmul(x, W)
        return super(Dense, self).build_graph(x, shape, tr, activation)


class Flatten(subgraph.Subgraph):
    def build_graph(self, x):
        return graph.Reshape(x, (-1, np.prod(x.node.shape.as_list()[1:]))).node


class Convolutions(subgraph.Subgraph):
    BORDER = {}
    ACTIVATION = {}

    def build_graph(self, x, convolutions):
        weights = []
        last = x
        for conv in convolutions:
            last = Convolution(last, **self._parse(conv.copy()))
            weights.append(last.weight)
        self.weight = graph.Variables(*weights)
        return last.node

    def _parse(self, conv):
        for key, mapping in [('border', self.BORDER),
                ('activation', self.ACTIVATION)]:
            if key in conv:
                conv[key] = mapping[conv[key]]
        return conv


class Input(subgraph.Subgraph):
    def build_graph(self, input):
        self.state = graph.Placeholder(np.float32,
                shape=[None] + input.shape + [input.history])

        convolutions = []
        if hasattr(input, 'use_convolutions'):
            convolutions = input.use_convolutions 
        conv = Convolutions(self.state, convolutions)

        self.weight = conv.weight
        return conv.node
