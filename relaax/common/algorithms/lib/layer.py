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

    @staticmethod
    def Softplus(x):
        return tf.nn.softplus(x)


class Border(object):
    Valid = 'VALID'
    Same = 'SAME'


class BaseLayer(subgraph.Subgraph):
    def build_graph(self, x, shape, transformation, activation):
        d = 1.0
        p = np.prod(shape[:-1])
        if p != 0:
            d = 1.0 / np.sqrt(p)
        initializer = graph.RandomUniformInitializer(minval=-d, maxval=d)
        W = graph.Variable(initializer(np.float32, shape)).node
        b = graph.Variable(initializer(np.float32, shape[-1:])).node
        self.weight = graph.TfNode((W, b))
        return activation(transformation(x.node, W) + b)


class Convolution(BaseLayer):
    def build_graph(self, x, n_filters, filter_size, stride,
            border=Border.Valid, activation=Activation.Null):
        shape = filter_size + [x.node.shape.as_list()[-1], n_filters]
        tr = lambda x, W: tf.nn.conv2d(x, W, strides=[1] + stride + [1],
                    padding=border)
        return super(Convolution, self).build_graph(x, shape, tr, activation)


class Dense(BaseLayer):
    def build_graph(self, x, size=1, activation=Activation.Null):
        assert len(x.node.shape) == 2
        shape = (x.node.shape.as_list()[1], size)
        tr = lambda x, W: tf.matmul(x, W)
        return super(Dense, self).build_graph(x, shape, tr, activation)


class LSTM(subgraph.Subgraph):
    def build_graph(self, x, batch_size=1, size=256):
        self.ph_step= graph.Placeholder(np.int32, [batch_size])

        self.ph_state = graph.TfNode(tuple(graph.Placeholder(np.float32, [batch_size, size]).node
                for _ in range(2)))

        self.zero_state = tuple(np.zeros([batch_size, size]) for _ in range(2))

        state = tf.contrib.rnn.LSTMStateTuple(*self.ph_state.node)

        lstm = tf.contrib.rnn.BasicLSTMCell(size, state_is_tuple=True)

        with tf.variable_scope('LSTM') as scope:
            outputs, self.state = tf.nn.dynamic_rnn(lstm, x.node,
                    initial_state=state, sequence_length=self.ph_step.node,
                    time_major=False, scope=scope)
            self.state = graph.TfNode(self.state)
            scope.reuse_variables()
            self.weight = graph.Variables(
                    graph.TfNode(tf.get_variable('basic_lstm_cell/weights')),
                    graph.TfNode(tf.get_variable('basic_lstm_cell/biases')))

        return outputs


class Flatten(subgraph.Subgraph):
    def build_graph(self, x):
        return graph.Reshape(x, (-1, np.prod(x.node.shape.as_list()[1:]))).node


class GenericLayers(subgraph.Subgraph):
    def build_graph(self, x, descs):
        weights = []
        last = x
        for desc in descs:
            props = desc.copy()
            del props['type']
            last = desc['type'](last, **props)
            weights.append(last.weight)
        self.weight = graph.Variables(*weights)
        return last.node


class DescreteActor(subgraph.Subgraph):
    def build_graph(self, head, action_size):
        actor = Dense(head, action_size, activation=Activation.Softmax)
        self.weight = actor.weight
        self.action_size = action_size
        self.continuous = False
        return actor.node


class ContinuousActor(subgraph.Subgraph):
    def build_graph(self, head, action_size):
        self.mu = Dense(head, action_size)
        self.sigma2 = Dense(head, action_size, activation=Activation.Softplus)
        self.weight = graph.Variables(self.mu.weight, self.sigma2.weight)
        self.action_size = action_size
        self.continuous = True
        return self.mu.node, self.sigma2.node


def Actor(head, output):
    Actor = ContinuousActor if output.continuous else DescreteActor
    return Actor(head, output.action_size)


class Input(subgraph.Subgraph):
    def build_graph(self, input, descs=None):
        input_shape = input.shape
        if np.prod(input.shape) == 0:
            input_shape = [1]
        shape = [None] + input_shape + [input.history]
        self.ph_state = graph.Placeholder(np.float32, shape=shape)

        if not input.use_convolutions or len(shape) <= 4:
            state_input = self.ph_state
        else:
            # move channels after history
            perm = list(range(len(shape)))
            perm = perm[0:3] + perm[-1:] + perm[3:-1]
            transpose = tf.transpose(self.ph_state.node, perm=perm)

            # mix history and channels in one dimension
            state_input = graph.TfNode(tf.reshape(transpose,
                [-1] + shape[1:3] + [np.prod(shape[3:])]))

        if input.use_convolutions and descs is None:
            # applying vanilla A3C convolution layers
            descs = [
                dict(type=Convolution, n_filters=16, filter_size=[8, 8],
                     stride=[4, 4], activation=Activation.Relu),
                dict(type=Convolution, n_filters=32, filter_size=[4, 4],
                     stride=[2, 2], activation=Activation.Relu)]

        descs = [] if not input.use_convolutions else descs
        layers = GenericLayers(state_input, descs)

        self.weight = layers.weight
        return layers.node


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
