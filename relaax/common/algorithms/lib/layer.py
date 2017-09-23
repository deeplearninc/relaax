from __future__ import division
from builtins import object
import numpy as np
import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import utils


class Activation(object):
    @classmethod
    def _make_map(cls):
        map = {}
        for name in ['Null', 'Relu', 'Elu', 'Sigmoid', 'Tanh', 'Softmax', 'Softplus']:
            activation = getattr(cls, name)
            map[name] = activation
            map[name.lower()] = activation
        return map

    @classmethod
    def get_activation(cls, name):
        return cls._map[name]

    @staticmethod
    def Null(x):
        return x

    @staticmethod
    def Relu(x):
        return tf.nn.relu(x)

    @staticmethod
    def Relu6(x):
        return tf.nn.relu6(x)

    @staticmethod
    def Elu(x):
        return tf.nn.elu(x)

    @staticmethod
    def Sigmoid(x):
        return tf.nn.sigmoid(x)

    @staticmethod
    def Tanh(x):
        return tf.nn.tanh(x)

    @staticmethod
    def Softmax(x):
        return tf.nn.softmax(x)

    @staticmethod
    def Softplus(x):
        return tf.nn.softplus(x)


Activation._map = Activation._make_map()


class Border(object):
    Valid = 'VALID'
    Same = 'SAME'


class BaseLayer(subgraph.Subgraph):
    def build_graph(self, x, shape, transformation, activation, d=None):
        if d is None:
            d = 1.0
            p = np.prod(shape[:-1])
            if p != 0:
                d = 1.0 / np.sqrt(p)
        initializer = graph.RandomUniformInitializer(minval=-d, maxval=d)
        W = graph.Variable(initializer(np.float32, shape)).node
        b = graph.Variable(initializer(np.float32, shape[-1:])).node
        self.weight = graph.TfNode((W, b))
        return activation(transformation(x, W) + b)


class Convolution(BaseLayer):
    def build_graph(self, x, n_filters, filter_size, stride,
                    border=Border.Valid, activation=Activation.Null):
        shape = filter_size + [x.node.shape.as_list()[-1], n_filters]

        def tr(x, W):
            return tf.nn.conv2d(x.node, W, strides=[1] + stride + [1], padding=border)

        return super(Convolution, self).build_graph(x, shape, tr, activation)


class Dense(BaseLayer):
    def build_graph(self, x, size=1, activation=Activation.Null, init_var=None):
        assert len(x.node.shape) >= 2
        shape = (x.node.shape.as_list()[-1], size)

        def tr(x, W):
            return tf.matmul(x.node, W)

        return super(Dense, self).build_graph(x, shape, tr, activation, d=init_var)


class DoubleDense(BaseLayer):
    def build_graph(self, x1, x2, size=1, activation=Activation.Null):
        assert len(x1.node.shape) == 2
        shape1 = (x1.node.shape.as_list()[1], size)
        assert len(x2.node.shape) == 2
        shape2 = (x2.node.shape.as_list()[1], size)

        d = 1.0
        p = np.prod(shape1[:-1])
        if p != 0:
            d = 1.0 / np.sqrt(p)
        initializer = graph.RandomUniformInitializer(minval=-d, maxval=d)
        W1 = graph.Variable(initializer(np.float32, shape1)).node

        d = 1.0
        p = np.prod(shape2[:-1])
        if p != 0:
            d = 1.0 / np.sqrt(p)
        initializer = graph.RandomUniformInitializer(minval=-d, maxval=d)
        W2 = graph.Variable(initializer(np.float32, shape2)).node

        initializer = graph.RandomUniformInitializer()
        b = graph.Variable(initializer(np.float32, shape2[-1:])).node
        self.weight = graph.TfNode((W1, W2, b))

        return activation(tf.matmul(x1.node, W1) + tf.matmul(x2.node, W2) + b)


class LSTM(subgraph.Subgraph):
    def build_graph(self, x, batch_size=1, n_units=256):
        self.phs = [graph.Placeholder(np.float32, [batch_size, n_units]) for _ in range(2)]
        self.ph_state = graph.TfNode(tuple(ph.node for ph in self.phs))
        self.ph_state.checked = tuple(ph.checked for ph in self.phs)

        self.zero_state = tuple(np.zeros([batch_size, n_units]) for _ in range(2))

        state = tf.contrib.rnn.LSTMStateTuple(*self.ph_state.checked)

        lstm = tf.contrib.rnn.BasicLSTMCell(n_units, state_is_tuple=True)

        outputs, self.state = tf.nn.dynamic_rnn(lstm, x.node, initial_state=state,
                                                sequence_length=tf.shape(x.node)[1:2], time_major=False)

        self.state = graph.TfNode(self.state)
        self.weight = graph.TfNode(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                     tf.get_variable_scope().name))
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


class DiscreteActor(subgraph.Subgraph):
    def build_graph(self, head, output):
        action_size = output.action_size
        actor = Dense(head, action_size, activation=Activation.Softmax)
        self.weight = actor.weight
        self.action_size = action_size
        self.continuous = False
        return actor.node


class ContinuousActor(subgraph.Subgraph):
    def build_graph(self, head, output):
        action_size = output.action_size
        self.mu = Dense(head, action_size, activation=Activation.Tanh)
        self.sigma2 = Dense(head, action_size, activation=Activation.Softplus)
        self.weight = graph.Variables(self.mu.weight, self.sigma2.weight)
        self.action_size = action_size
        self.continuous = True
        return self.mu.node * output.scale, self.sigma2.node + 1e-4


def Actor(head, output):
    Actor = ContinuousActor if output.continuous else DiscreteActor
    return Actor(head, output)


class DDPGActor(subgraph.Subgraph):
    def build_graph(self, head, output):
        self.action_size = output.action_size
        self.continuous = True

        self.out = Dense(head, self.action_size, activation=Activation.Tanh, init_var=3e-3)
        self.weight = self.out.weight
        self.scaled_out = graph.TfNode(self.out.node * output.scale)


class InputPlaceholder(subgraph.Subgraph):
    def build_graph(self, input):
        if hasattr(input, 'shape'):
            input_shape = input.shape
        else:
            input_shape = input.image

        if np.prod(input_shape) == 0:
            input_shape = [1]
        shape = [None] + input_shape + [input.history]
        self.ph_state = graph.Placeholder(np.float32, shape=shape)

        if not input.use_convolutions or len(shape) <= 4:
            state_input = self.ph_state.checked
        else:
            # move channels after history
            perm = list(range(len(shape)))
            perm = perm[0:3] + perm[-1:] + perm[3:-1]
            transpose = tf.transpose(self.ph_state.checked, perm=perm)

            # mix history and channels in one dimension
            state_input = tf.reshape(transpose, [-1] + shape[1:3] + [np.prod(shape[3:])])

        return state_input


class Input(subgraph.Subgraph):
    def build_graph(self, input, descs=None, input_placeholder=None):
        if input_placeholder is None:
            input_placeholder = InputPlaceholder(input)
        self.ph_state = input_placeholder.ph_state

        if input.use_convolutions and descs is None:
            # applying vanilla A3C convolution layers
            descs = [dict(type=Convolution, n_filters=16, filter_size=[8, 8], stride=[4, 4],
                          activation=Activation.Relu),
                     dict(type=Convolution, n_filters=32, filter_size=[4, 4], stride=[2, 2],
                          activation=Activation.Relu)]

        descs = [] if not input.use_convolutions else descs
        layers = GenericLayers(input_placeholder, descs)

        self.weight = layers.weight
        return layers.node


class Weights(subgraph.Subgraph):
    def build_graph(self, *layers):
        weights = [layer.weight.node for layer in layers]
        self.ph_weights = graph.Placeholders(variables=graph.TfNode(weights))
        self.assign = graph.TfNode([tf.assign(variable, value) for variable, value in
                                    utils.Utils.izip(weights, self.ph_weights.checked)])
        self.check = graph.TfNode(tf.group(*[tf.check_numerics(w, 'weight_%d' % i) for i, w in
                                             enumerate(utils.Utils.flatten(weights))]))
        self.global_norm = tf.global_norm(list(utils.Utils.flatten(weights)))
        return weights
