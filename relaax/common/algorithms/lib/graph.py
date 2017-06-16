from __future__ import print_function
from builtins import object
import tensorflow as tf
from tensorflow.python.ops import init_ops
import numpy as np

from relaax.common.algorithms.lib import utils
from relaax.common.algorithms import subgraph


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


class AdamOptimizer(subgraph.Subgraph):
    def build_graph(self, learning_rate=0.001):
        return tf.train.AdamOptimizer(learning_rate=learning_rate)


class RMSPropOptimizer(subgraph.Subgraph):
    def build_graph(self, learning_rate, decay, momentum, epsilon):
        return tf.train.RMSPropOptimizer(
            learning_rate=learning_rate.node,
            decay=decay,
            momentum=momentum,
            epsilon=epsilon
        )


class L2loss(subgraph.Subgraph):
    """Computes half the L2 norm of a tensor without the sqrt."""

    def build_graph(self, t, name=None):
        """
        Args:
            t: A Tensor.
            name: A name for the operation (optional).

        Returns:
            A Tensor. Has the same type as t.
        """

        return tf.nn.l2_loss(t, name=name)


class Softmax(subgraph.Subgraph):
    def build_graph(self, x):
        return tf.nn.softmax(x.node)


class Reshape(subgraph.Subgraph):
    def build_graph(self, x, shape):
        return tf.reshape(x.node, shape)


class Flatten(subgraph.Subgraph):
    def build_graph(self, x):
        return Reshape(x, (-1, )).node


class Expand(subgraph.Subgraph):
    def build_graph(self, x, dim):
        return tf.expand_dims(x.node, dim)


class Concat(subgraph.Subgraph):
    def build_graph(self, concat_dim, values, name='concat'):
        return tf.concat(concat_dim, [v.node for v in values], name=name)


class List(subgraph.Subgraph):
    def build_graph(self, items):
        self.items = list(items)
        return [i.node for i in self.items]


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


class VarAssign(subgraph.Subgraph):
    def build_graph(self, variable, value):
        self.ph_variable = Placeholders(variables=TfNode(variable))
        self.assign_from_ph = TfNode(tf.assign(variable, self.ph_variable.node))
        self.assign_from_value = TfNode(tf.assign(variable, tf.constant(value)))
        return variable


class Constant(subgraph.Subgraph):
    """Creates a constant tensor."""

    def build_graph(self, value, dtype=None, shape=None, name='Const'):
        """
        Args:
            value: A constant value (or list) of output type dtype.
            dtype: The type of the elements of the resulting tensor.
            shape: Optional dimensions of resulting tensor.
            name: Optional name for the tensor.

        Returns:
            A Constant Tensor.
        """

        return tf.constant(value, dtype=dtype, shape=shape, name=name)


class Placeholder(subgraph.Subgraph):
    """Placeholder of given shape."""

    DTYPE = {
        np.int32: tf.int32,
        np.int64: tf.int64,
        np.float32: tf.float32
    }

    def build_graph(self, dtype, shape=None, name=None):
        """Assemble one placeholder.

        Args:
            shape: placeholder shape
            dtype: placeholder data type
            name: placeholder name

        Returns:
            placeholder of given shape and data type
        """

        return tf.placeholder(self.DTYPE[dtype], shape=shape, name=name)


class Placeholders(subgraph.Subgraph):
    def build_graph(self, variables):
        return utils.Utils.map(
            variables.node,
            lambda v: tf.placeholder(shape=v.get_shape(), dtype=v.dtype)
        )


class GlobalStep(subgraph.Subgraph):
    def build_graph(self):
        self.n = Variable(0, dtype=np.int64)
        self.ph_increment = Placeholder(np.int64)
        self.increment = Increment(self.n, self.ph_increment)


class Variable(subgraph.Subgraph):
    DTYPE = {
        None: None,
        np.int64: tf.int64,
        np.float32: tf.float32,
        np.float64: tf.float64
    }

    def build_graph(self, initial_value, dtype=None, name=None):
        return tf.Variable(initial_value, dtype=self.DTYPE[dtype], name=name)


class Variables(subgraph.Subgraph):
    def build_graph(self, *variables):
        return [variable.node for variable in variables]

    def assign(self, values):
        return TfNode([
            tf.assign(variable, value)
            for variable, value in utils.Utils.izip(
                self.node,
                values.node
            )
        ])


class Initialize(subgraph.Subgraph):
    def build_graph(self):
        return tf.global_variables_initializer()


class TfNode(subgraph.Subgraph):
    def build_graph(self, tf_tensor):
        return tf_tensor
