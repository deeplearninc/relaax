from __future__ import division
from builtins import object
import collections
import numpy as np
import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import utils


class NullSubgraph(subgraph.Subgraph):
    def build_graph(self, x):
        self.weight = []
        return x


class SigmoidSubgraph(subgraph.Subgraph):
    def build_graph(self, x):
        self.weight = []
        return tf.nn.sigmoid(x)


class TanhSubgraph(subgraph.Subgraph):
    def build_graph(self, x):
        self.weight = []
        return tf.nn.tanh(x)


class ReluSubgraph(subgraph.Subgraph):
    def build_graph(self, x):
        self.weight = []
        return tf.nn.relu(x)


class Relu6Subgraph(subgraph.Subgraph):
    def build_graph(self, x):
        self.weight = []
        return tf.nn.relu6(x)


class EluSubgraph(subgraph.Subgraph):
    def build_graph(self, x):
        self.weight = []
        return tf.nn.elu(x)


class SoftmaxSubgraph(subgraph.Subgraph):
    def build_graph(self, x):
        self.weight = []
        return tf.nn.softmax(x)


class SoftplusSubgraph(subgraph.Subgraph):
    def build_graph(self, x):
        self.weight = []
        return tf.nn.softplus(x)


class KAFConfigurator(object):
    def __init__(self, cfg):
        self.D = tf.linspace(start=-cfg.boundary, stop=-cfg.boundary, num=cfg.size)
        self.kernel = cfg.kernel
        self.gamma = cfg.gamma

    def __call__(self, x):
        return KAFSubgraph(x, self.D, self.kernel, self.gamma)


class Activation(object):
    Null = NullSubgraph
    Sigmoid = SigmoidSubgraph
    Tanh = TanhSubgraph
    Relu = ReluSubgraph
    Relu6 = Relu6Subgraph
    Elu = EluSubgraph
    Softmax = SoftmaxSubgraph
    Softplus = SoftplusSubgraph
    KAF = KAFConfigurator

    # dict comprehension has its own locals. dict comprehension does not see class context.
    # This is why class locals are passed as lambda argument
    _MAP = (lambda locals_: {k.lower(): k for k in locals_.keys() if not k.startswith('_')})(locals())

    @classmethod
    def get_activation(cls, name):
        return getattr(cls, cls._MAP[name.lower()])


def get_activation(cfg):
    activation = Activation.get_activation(cfg.activation)
    if cfg.activation.lower() == 'kaf':
        activation = activation(cfg.KAF)
    return activation


class KAFSubgraph(subgraph.Subgraph):
    def build_graph(self, x, d, kernel='rbf', gamma=1.):
        initializer = graph.RandomNormalInitializer(stddev=0.1)

        if kernel == 'rbf':
            k = gauss_kernel(x, d, gamma=gamma)

            shape = (1, x.get_shape().as_list()[-1], d.get_shape().as_list()[0])
            alpha = graph.Variable(initializer(np.float32, shape=shape))

        elif kernel == 'rbf2d':
            dx, dy = tf.meshgrid(d, d)
            k = gauss_kernel2d(x, dx, dy, gamma=gamma)

            shape = (1, x.get_shape().as_list()[-1] // 2,
                     d.get_shape().as_list()[0] * d.get_shape().as_list()[0])
            alpha = graph.Variable(initializer(np.float32, shape=shape))
        else:
            assert False, "There are 2 valid options for KAF kernels: rbf | rbf2d"

        self.weight = alpha.node
        return tf.reduce_sum(tf.multiply(k, self.weight), axis=-1)


def gauss_kernel(x, d, gamma=1.):
    x = tf.expand_dims(x, axis=-1)
    if x.get_shape().ndims < 4:
        d = tf.reshape(d, (1, 1, -1))
    else:
        d = tf.reshape(d, (1, 1, 1, 1, -1))

    return tf.exp(- gamma * tf.square(x - d))


def gauss_kernel2d(x, dx, dy, gamma=1.):
    h_size = x.get_shape()[-1].value // 2

    x = tf.expand_dims(x, axis=-1)
    if x.get_shape().ndims < 4:
        dx = tf.reshape(dx, (1, 1, -1))
        dy = tf.reshape(dy, (1, 1, -1))
        x1, x2 = x[:, :h_size], x[:, h_size:]
    else:
        dy = tf.reshape(dy, (1, 1, 1, 1, -1))
        dx = tf.reshape(dx, (1, 1, 1, 1, -1))
        x1, x2 = x[:, :, :, :h_size], x[:, :, :, h_size:]

    return tf.exp(-gamma * tf.square(x1 - dx)) + tf.exp(- gamma * tf.square(x2 - dy))


class Border(object):
    Valid = 'VALID'
    Same = 'SAME'

    # dict comprehension has its own locals. dict comprehension does not see class context.
    # This is why class locals are passed as lambda argument
    _MAP = (lambda locals_: {k.lower(): k for k in locals_.keys() if not k.startswith('_')})(locals())

    @classmethod
    def get_border(cls, name):
        return getattr(cls, cls._MAP[name.lower()])


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
        activation = activation(transformation(x, W) + b)
        self.weight = graph.TfNode((W, b, activation.weight))
        return activation.node


class Convolution(BaseLayer):
    def build_graph(self, x, n_filters=32, filter_size=[3, 3], stride=[2, 2], border=Border.Same,
                    activation=Activation.Elu):
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

        activation = activation(tf.matmul(x1.node, W1) + tf.matmul(x2.node, W2) + b)
        self.weight = graph.TfNode((W1, W2, b, activation.weight))
        return activation.node


def lstm(lstm_type, x, batch_size=1, n_units=256, n_cores=8):
    if lstm_type.lower() == 'basic':
        return LSTM(x, batch_size, n_units)
    elif lstm_type.lower() == 'dilated':
        return DilatedLSTM(x, batch_size, n_units, n_cores)
    else:
        assert False, "There are 2 valid options for LSTM type: Basic | Dilated"


class LSTM(subgraph.Subgraph):
    def build_graph(self, x, batch_size, n_units):
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
        self.reset_timestep = None
        return outputs


class DilatedLSTM(subgraph.Subgraph):
    def build_graph(self, x, batch_size, n_units, n_cores):
        lstm = graph.DilatedLSTMCell(n_units, n_cores)

        self.ph_state = graph.Placeholder(np.float32, [batch_size, lstm.state_size])
        self.zero_state = np.zeros([batch_size, lstm.state_size])

        outputs, self.state = tf.nn.dynamic_rnn(lstm, x.node, initial_state=self.ph_state.checked,
                                                sequence_length=tf.shape(x.node)[1:2], time_major=False)

        self.state = graph.TfNode(self.state)
        self.weight = graph.TfNode([lstm.matrix, lstm.bias])
        self.reset_timestep = graph.TfNode(lstm.reset_timestep)
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
        return self.mu.node * output.scale, self.sigma2.node + tf.constant(1e-8)


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

        if len(shape) <= 4:
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
    def build_graph(self, input, descs, input_placeholder=None):
        if input_placeholder is None:
            input_placeholder = InputPlaceholder(input)
        self.ph_state = input_placeholder.ph_state

        layers = GenericLayers(input_placeholder, descs)

        self.weight = layers.weight
        return layers.node


class Type(object):
    _MAP = {name.lower(): globals()[name] for name in ['Convolution', 'Dense', 'Flatten']}

    @classmethod
    def get_type(cls, name):
        return cls._MAP[name.lower()]


class ConfiguredInput(subgraph.Subgraph):
    _MAP = collections.defaultdict(lambda: lambda x: x, type=Type.get_type,
                                   activation=Activation.get_activation, border=Border.get_border)

    def build_graph(self, input, input_placeholder=None):
        if hasattr(input, 'layers'):
            descs = self.read_layers(input.layers)
        else:
            if input.use_convolutions:
                if input.universe:
                    conv_layer = dict(type=Convolution, activation=Activation.Elu, n_filters=32,
                                      filter_size=[3, 3], stride=[2, 2], border=Border.Same)
                    descs = [conv_layer] * 4
                else:
                    descs = [dict(type=Convolution, n_filters=16, filter_size=[8, 8], stride=[4, 4],
                                  activation=Activation.Relu),
                             dict(type=Convolution, n_filters=32, filter_size=[4, 4], stride=[2, 2],
                                  activation=Activation.Relu)]
            else:
                descs = []

        input = Input(input, descs, input_placeholder=input_placeholder)
        self.ph_state = input.ph_state
        self.weight = input.weight
        return input.node

    def read_layers(self, layers):
        return [self.read_layer(layer) for layer in layers]

    def read_layer(self, layer):
        return {k: self._MAP[k](v) for k, v in layer.items()}


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
