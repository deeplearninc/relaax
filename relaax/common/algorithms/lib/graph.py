import tensorflow as tf
from tensorflow.python.ops import init_ops
import numpy as np

from relaax.common.algorithms.lib import utils
from relaax.common.algorithms import subgraph


class Convolution(subgraph.Subgraph):
    def build_graph(self, x, wb, stride):
        return tf.nn.conv2d(x.node, wb.W, strides=[1, stride, stride, 1], padding="VALID") + wb.b


class Relu(subgraph.Subgraph):
    def build_graph(self, x):
        return tf.nn.relu(x.node)


class Reshape(subgraph.Subgraph):
    def build_graph(self, x, shape):
        return tf.reshape(x.node, shape)


class Softmax(subgraph.Subgraph):
    def build_graph(self, x):
        return tf.nn.softmax(x.node)


class List(subgraph.Subgraph):
    def build_graph(self, nodes):
        return map(lambda n: n.node, nodes)


class Assign(subgraph.Subgraph):
    def build_graph(self, variables, values):
        return [
            tf.assign(variable, value)
            for variable, value in utils.Utils.izip(
                variables.node,
                values.node
            )
        ]


class Increment(subgraph.Subgraph):
    def build_graph(self, variable, increment):
        return tf.assign_add(variable.node, increment.node)


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


class GlobalStep(subgraph.Subgraph):
    def build_graph(self, increment):
        self.n = Variable(0, dtype=np.int64)
        self.increment = Increment(self.n, increment)


class Variable(subgraph.Subgraph):
    DTYPE = {
        None: None,
        np.int64: tf.int64
    }

    def build_graph(self, initial_value, dtype=None):
        return tf.Variable(initial_value, dtype=self.DTYPE[dtype])


class VariablesByShapes(subgraph.Subgraph):
    """Holder for variables representing weights of the fully connected NN."""

    def build_graph(self, dtype, shapes, initializer=DefaultInitializer()):
        return [
            tf.Variable(initial_value=initializer(shape=shape, dtype=dtype))
            for shape in shapes
        ]


class Wb(subgraph.Subgraph):
    def build_graph(self, dtype, shape):
        d = 1.0 / np.sqrt(np.prod(shape[:-1]))
        initializer = RandomUniformInitializer(minval=-d, maxval=d)
        self.W = Variable(initializer(dtype, shape     )).node
        self.b = Variable(initializer(dtype, shape[-1:])).node
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


class Gradients(subgraph.Subgraph):
    def build_graph(self, loss, variables):
        return utils.Utils.reconstruct(
            tf.gradients(loss.node, list(utils.Utils.flatten(variables.node))),
            variables.node
        )


class ApplyGradients(subgraph.Subgraph):
    def build_graph(self, optimizer, weights, gradients, global_step):
        self.apply_gradients_op = optimizer.node.apply_gradients(
            utils.Utils.izip(gradients.node, weights.node)
        )
        return tf.group(
            self.apply_gradients_op,
            global_step.increment.node
        )


class AdamOptimizer(subgraph.Subgraph):
    def build_graph(self, learning_rate=0.001):
        return tf.train.AdamOptimizer(learning_rate=learning_rate)


class Initialize(subgraph.Subgraph):
    def build_graph(self):
        return tf.global_variables_initializer()
