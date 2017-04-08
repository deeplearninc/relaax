import tensorflow as tf
from tensorflow.python.ops import init_ops
import numpy as np

from relaax.common.algorithms import subgraph


class DefaultInitializer(object):
    MAP = {
        np.float32: (tf.float32, init_ops.glorot_uniform_initializer)
    }

    def __call__(self, shape=None, dtype=np.float32):
        tf_dtype, initializer = self.MAP[dtype]
        return initializer(dtype=tf_dtype)(shape=shape, dtype=tf_dtype)


class ZeroInitializer(object):
    def __call__(self, shape=None, dtype=np.float):
        return np.zeros(shape=shape, dtype=dtype)


class XavierInitializer(object):
    DTYPES = {
        np.float: tf.float64,
        np.float64: tf.float64,
        np.float32: tf.float32,
    }

    def __call__(self, shape=None, dtype=np.float):
        return tf.contrib.layers.xavier_initializer()(
            shape=shape,
            dtype=self.DTYPES[dtype]
        )


class Placeholder(subgraph.Subgraph):
    """Placeholder of given shape."""

    def build_graph(self, shape, dtype=np.float32):
        """Assemble one placeholder.

        Args:
            shape: placehoder shape
            dtype: placeholder data type

        Returns:
            placeholder of given shape and data type
        """

        return tf.placeholder(shape=shape, dtype=dtype)


class Placeholders(subgraph.Subgraph):
    """List of placeholders of given shapes."""

    def build_graph(self, shapes):
        """Assemble list of placeholders.

        Args:
            shapes: defines shape for placeholders, dtype will be np.float32

        Returns:
            list of placeholders
        """

        return [tf.placeholder(np.float32, shape) for shape in shapes]


class Variables(subgraph.Subgraph):
    """Holder for variables representing weights of the fully connected NN."""

    DTYPES = {
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
                    dtype=self.DTYPES[p.dtype]
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


class FullyConnected(subgraph.Subgraph):
    """Builds fully connected neural network."""

    def build_graph(self, state, weights):
        self.weights = weights
        last = state.node
        for w in weights.node:
            last = tf.nn.relu(tf.matmul(last, w))
        return tf.nn.softmax(last)


class PolicyLoss(subgraph.Subgraph):
    def build_graph(self, action, network, discounted_reward):
        # making actions that gave good advantage (reward over time) more likely,
        # and actions that didn't less likely.

        self.log_like_op = tf.log(tf.reduce_sum(action.node * network.node, axis=[1]))
        self.production_op = self.log_like_op * discounted_reward.node
        return -tf.reduce_sum(self.production_op)

    def log_like(self, state, action):
        return subgraph.Subgraph.Op(self.log_like_op, state=state, action=action)

    def production(self, state, action, discounted_reward):
        return subgraph.Subgraph.Op(
            self.production_op,
            state=state,
            action=action,
            discounted_reward=discounted_reward
        )


class Policy(subgraph.Subgraph):
    def build_graph(self, network, loss):
        self.gradients = tf.gradients(loss.node, network.weights.node)
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
            zip(gradients.node, weights.node)
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
