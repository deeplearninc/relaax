import tensorflow as tf
import numpy as np

from relaax.common.algorithms.subgraph import Subgraph
from ..pg_config import config


class Zero(object):
    def __call__(self, shape=None, dtype=np.float):
        return np.zeros(shape=shape, dtype=dtype)


class Xavier(object):
    DTYPES = {
        np.float: tf.float64,
        np.float64: tf.float64,
        np.float32: tf.float32
    }

    def __call__(self, shape, dtype=np.float):
        return tf.contrib.layers.xavier_initializer()(
            shape=shape,
            dtype=self.DTYPES[dtype]
        )


class Placeholder(Subgraph):
    def build(self, shape, dtype=np.float32):
        return tf.placeholder(shape=shape, dtype=dtype)


class Placeholders(Subgraph):
    def build(self, variables=[]):
        return [
            tf.placeholder(v.dtype, v.get_shape())
            for v in variables.node
        ]


class Assign(Subgraph):
    def build(self, variables, values):
        return [
            tf.assign(variable, value)
            for variable, value in zip(variables.node, values.node)
        ]


class Weights(Subgraph):
    """Holder for variables representing weights of the fully connected NN."""

    def build(self, initializer=Zero()):
        """Assemble weights of the NN into tf graph.

        Args:
            shapes: sizes of the weights variables
            initializer: initializer for variables

        Returns:
            list to the 'weights' tensors in the graph

        """

        state_size=config.state_size
        hidden_sizes=config.hidden_layers
        action_size=config.action_size

        shapes = zip([state_size] + hidden_sizes, hidden_sizes + [action_size])
        return [
            tf.Variable(initial_value=initializer(shape=shape, dtype=np.float32))
            for shape in shapes
        ]


class FullyConnected(Subgraph):
    """Builds fully connected neural network."""

    def build(self, state, weights):
        last = state.node
        for w in weights.node:
            last = tf.nn.relu(tf.matmul(last, w))
        return tf.nn.softmax(last)


class Loss(Subgraph):
    def build(self, action, policy, discounted_reward):
        # making actions that gave good advantage (reward over time) more likely,
        # and actions that didn't less likely.

        log_like = tf.log(tf.reduce_sum(action.node * policy.node))
        return -tf.reduce_mean(log_like * discounted_reward.node)


class ApplyGradients(Subgraph):
    def build(self, optimizer, gradients, weights):
        return optimizer.node.apply_gradients(
            zip(gradients.node, weights.node)
        )


class Adam(Subgraph):
    def build(self, learning_rate=0.001):
        return tf.train.AdamOptimizer(learning_rate=learning_rate)


class Gradients(Subgraph):
    def build(self, loss, variables):
        pass


class Initialize(Subgraph):
    def build(self):
        return tf.global_variables_initializer()
