import tensorflow as tf
from tensorflow.python.ops import init_ops
import numpy as np

from relaax.common.algorithms.lib import utils
from relaax.common.algorithms import subgraph


class List(subgraph.Subgraph):
    def build_graph(self, nodes):
        return map(lambda n: n.node, nodes)

    def get(self):
        return subgraph.Subgraph.Op(self.node)


class Assign(subgraph.Subgraph):
    def build_graph(self, variables, values):
        return [
            tf.assign(variable, value)
            for variable, value in utils.Utils.izip(
                variables.node,
                values.node
            )
        ]

    def assign(self, values):
        return subgraph.Subgraph.Op(self.node, values=values)


class Counter(subgraph.Subgraph):
    DTYPE = {np.int64: tf.int64}

    def build_graph(self, inc_value, dtype=np.int64):
        self.counter = tf.Variable(0, dtype=self.DTYPE[dtype])
        self.increment = tf.assign_add(self.counter, inc_value.node)

    def value(self):
        return subgraph.Subgraph.Op(self.counter)


class DefaultInitializer(object):
    INIT = {
        np.float32: (tf.float32, init_ops.glorot_uniform_initializer)
    }

    def __call__(self, dtype=np.float32, shape=None):
        tf_dtype, initializer = self.INIT[dtype]
        return initializer(dtype=tf_dtype)(shape=shape, dtype=tf_dtype)


class ZeroInitializer(object):
    def __call__(self, dtype=np.float32, shape=None):
        return np.zeros(shape=shape, dtype=dtype)


class OneInitializer(object):
    def __call__(self, dtype=np.float32, shape=None):
        return np.ones(shape=shape, dtype=dtype)


class RandomUniformInitializer(object):
    DTYPE = {
        np.float: tf.float64,
        np.float64: tf.float64,
        np.float32: tf.float32,
    }

    def __init__(self, minval=0, maxval=1):
        self.minval = minval
        self.maxval = maxval

    def __call__(self, dtype=np.float32, shape=None):
        return tf.random_uniform(
            shape,
            dtype=self.DTYPE[dtype],
            minval=self.minval,
            maxval=self.maxval
        )


class XavierInitializer(object):
    DTYPE = {
        np.float: tf.float64,
        np.float64: tf.float64,
        np.float32: tf.float32,
    }

    def __call__(self, dtype=np.float32, shape=None):
        return tf.contrib.layers.xavier_initializer()(
            dtype=self.DTYPE[dtype],
            shape=shape
        )


class Placeholder(subgraph.Subgraph):
    """Placeholder of given shape."""

    DTYPE = {
        np.int32: tf.int32,
        np.int64: tf.int64,
        np.float32: tf.float32
    }

    def build_graph(self, dtype, shape=None):
        """Assemble one placeholder.

        Args:
            shape: placehoder shape
            dtype: placeholder data type

        Returns:
            placeholder of given shape and data type
        """

        return tf.placeholder(self.DTYPE[dtype], shape=shape)


class PlaceholdersByVariables(subgraph.Subgraph):
    def build_graph(self, variables):
        return utils.Utils.map(
            variables.node,
            lambda v: tf.placeholder(shape=v.get_shape(), dtype=v.dtype)
        )


class PlaceholdersByShapes(subgraph.Subgraph):
    def build_graph(self, dtype, ):
        return utils.Utils.map(
            variables.node,
            lambda v: tf.placeholder(shape=v.get_shape(), dtype=v.dtype)
        )


class Placeholders(subgraph.Subgraph):
    """List of placeholders of given shapes."""

    def build_graph(self, shapes):
        """Assemble list of placeholders.

        Args:
            shapes: defines shape for placeholders, dtype will be np.float32

        Returns:
            list of placeholders
        """

        def pairs(shapes):
            for shape in shapes:
                yield shape
                yield shape[-1]

        return [tf.placeholder(np.float32, shape) for shape in pairs(shapes)]


class Variable(subgraph.Subgraph):
    def build_graph(self, dtype, shape, initializer=DefaultInitializer()):
        return tf.Variable(initial_value=initializer(dtype=dtype, shape=shape))


class VariablesByShapes(subgraph.Subgraph):
    """Holder for variables representing weights of the fully connected NN."""

    def build_graph(self, dtype, shapes, initializer=DefaultInitializer()):
        return [
            tf.Variable(initial_value=initializer(shape=shape, dtype=dtype))
            for shape in shapes
        ]

    def get(self):
        return subgraph.Subgraph.Op(self.node)

    def assign(self, values):
        return subgraph.Subgraph.Op(self.assign_op, values=values)


class Variables(subgraph.Subgraph):
    """Holder for variables representing weights of the fully connected NN."""

    DTYPE = {
        tf.float64: np.float64,
        tf.float32: np.float32,
    }

    def build_graph(self, placeholders=None, shapes=None, initializer=DefaultInitializer()):
        """Assemble list of variables.

        Args:
            placeholders: defines shape and type for variables
            shapes: defines shape for variables, dtype will be np.float32
            initializer: initializer for variables

        Returns:
            list to the 'weights' variables in the graph
        """
        
        if placeholders is None and shapes is None:
            raise RuntimeError('Neither placholders nor shapes parameters are supplied.')

        if placeholders is not None and shapes is not None:
            raise RuntimeError('Both placholders and shapes parameters are supplied.')

        if placeholders is not None:
            variables = [
                tf.Variable(initial_value=initializer(
                    shape=p.shape.as_list(),
                    dtype=self.DTYPE[p.dtype]
                ))
                for p in placeholders.node
            ]

            with tf.variable_scope('assign'):
                self.assign_op = [
                    tf.assign(variable, value)
                    for variable, value in zip(variables, placeholders.node)
                ]
        else:
            variables = [
                tf.Variable(initial_value=initializer(
                    shape=shape,
                    dtype=np.float32
                ))
                for shape in shapes
            ]

        return variables

    def get(self):
        return subgraph.Subgraph.Op(self.node)

    def assign(self, values):
        return subgraph.Subgraph.Op(self.assign_op, values=values)


class Wb(subgraph.Subgraph):
    def build_graph(self, dtype, shape):
        d = 1.0 / np.sqrt(np.prod(shape[:-1]))
        initializer = RandomUniformInitializer(minval=-d, maxval=d)
        self.W = Variable(dtype, shape     , initializer).node
        self.b = Variable(dtype, shape[-1:], initializer).node
        return self.W, self.b


class ApplyWb(subgraph.Subgraph):
    def build_graph(self, x, wb):
        return tf.matmul(x.node, wb.W) + wb.b


class FullyConnected(subgraph.Subgraph):
    """Builds fully connected neural network."""

    def build_graph(self, state, weights):
        self.weights = weights
        last = state.node
        for w, b in weights.node:
            last = tf.nn.relu(tf.matmul(last, w) + b)
        return tf.nn.softmax(last)


class PolicyLoss(subgraph.Subgraph):
    def build_graph(self, action, action_size, network, discounted_reward):
        # making actions that gave good advantage (reward over time) more likely,
        # and actions that didn't less likely.

        log_like_op = tf.log(tf.reduce_sum(
            tf.one_hot(action.node, action_size) * network.node,
            axis=[1]
        ))
        return -tf.reduce_sum(log_like_op * discounted_reward.node)


class Policy(subgraph.Subgraph):
    def build_graph(self, network, loss):
        self.gradients = utils.Utils.reconstruct(
            tf.gradients(loss.node, list(utils.Utils.flatten(network.weights.node))),
            network.weights.node
        )
        return network.node

    def get_action(self, state):
        return subgraph.Subgraph.Op(self.node, state=state)

    def compute_gradients(self, state, action, discounted_reward):
        return subgraph.Subgraph.Op(
            self.gradients,
            state=state,
            action=action,
            discounted_reward=discounted_reward
        )


class ApplyGradients(subgraph.Subgraph):
    def build_graph(self, optimizer, weights, gradients):
        self.apply_gradients_op = optimizer.node.apply_gradients(
            utils.Utils.izip(gradients.node, weights.node)
        )

    def apply_gradients(self, gradients):
        return subgraph.Subgraph.Op(self.apply_gradients_op, gradients=gradients)


class AdamOptimizer(subgraph.Subgraph):
    def build_graph(self, learning_rate=0.001):
        return tf.train.AdamOptimizer(learning_rate=learning_rate)


class Initialize(subgraph.Subgraph):
    def build_graph(self):
        return tf.global_variables_initializer()

    def initialize(self):
        return subgraph.Subgraph.Op(self.node)
