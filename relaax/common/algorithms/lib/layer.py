from __future__ import division
from builtins import object
import numpy as np
import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import utils


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
        d = 1.0 / np.sqrt(np.prod(shape[:-1]))
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
        self.ph_state = graph.Placeholder(np.float32,
                shape=[None] + input.shape + [input.history])

        convolutions = []
        if input.use_convolutions:
            convolutions = [
                    dict(n_filters=16, filter_size=[8, 8], stride=[4, 4]),
                    dict(n_filters=32, filter_size=[4, 4], stride=[2, 2])]
        conv = Convolutions(self.ph_state, convolutions)

        self.weight = conv.weight
        return conv.node


class Weights(subgraph.Subgraph):
    def build_graph(self, *layers):
        weights = [layer.weight.node for layer in layers]
        self.ph_weights = graph.Placeholders(variables=graph.TfNode(weights))
        self.assign = graph.TfNode([tf.assign(variable, value)
                for variable, value in utils.Utils.izip(weights, self.ph_weights.node)])
        return weights


class Gradients(subgraph.Subgraph):
    def build_graph(self, weights, loss=None, optimizer=None):
        if loss is not None:
            self.calculate = graph.TfNode(utils.Utils.reconstruct(tf.gradients(
                loss.node, list(utils.Utils.flatten(weights.node))), weights.node))
        if optimizer is not None:
            self.ph_gradients = graph.Placeholders(weights)
            self.apply = graph.TfNode(optimizer.node.apply_gradients(
                    utils.Utils.izip(self.ph_gradients.node, weights.node)))
