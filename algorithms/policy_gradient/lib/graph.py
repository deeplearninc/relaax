import tensorflow as tf
import numpy as np

from relaax.common.algorithms.subgraph import Subgraph
from relaax.common.algorithms.op import Op
from ..pg_config import config


class Zero(object):
    DTYPES = {
        tf.float64: np.float64,
        tf.float32: np.float32,
        np.float32: np.float32
    }

    def __call__(self, shape=None, dtype=np.float):
        return np.zeros(shape=shape, dtype=self.DTYPES[dtype])


class Xavier(object):
    DTYPES = {
        np.float: tf.float64,
        np.float64: tf.float64,
        np.float32: tf.float32,
        tf.float32: tf.float32
    }

    def __call__(self, shape, dtype=np.float):
        return tf.contrib.layers.xavier_initializer()(
            shape=shape,
            dtype=self.DTYPES[dtype]
        )


class Placeholder(Subgraph):
    def build_graph(self, shape, dtype=np.float32):
        return tf.placeholder(shape=shape, dtype=dtype)


class Placeholders(Subgraph):
    def build_graph(self, shapes):
        return [
            tf.placeholder(np.float32, shape)
            for shape in shapes
        ]


class Variables(Subgraph):
    """Holder for variables representing weights of the fully connected NN."""

    def build_graph(self, tensors=None, shapes=None, initializer=Zero()):
        """Assemble  of the NN into tf graph.

        Args:
            shapes: sizes of the weights variables
            initializer: initializer for variables

        Returns:
            list to the 'weights' tensors in the graph

        """
        
        if tensors is not None:
            variables = [
                tf.Variable(initial_value=initializer(
                    shape=tensor.shape.as_list(),
                    dtype=tensor.dtype
                ))
                for tensor in tensors.node
            ]

            self.assign_op = [
                tf.assign(variable, value)
                for variable, value in zip(variables, tensors.node)
            ]

        if shapes is not None:
            variables = [
                tf.Variable(initial_value=initializer(
                    shape=shape,
                    dtype=np.float32
                ))
                for shape in shapes
            ]

        return variables

    def get(self):
        return Op(self.node)

    def assign(self, values):
        return Op(self.assign_op, values=values)


class FullyConnected(Subgraph):
    """Builds fully connected neural network."""

    def build_graph(self, state, weights):
        self.weights = weights
        last = state.node
        for w in weights.node:
            last = tf.nn.relu(tf.matmul(last, w))
        return tf.nn.softmax(last)


class PolicyLoss(Subgraph):
    def build_graph(self, action, network, discounted_reward):
        # making actions that gave good advantage (reward over time) more likely,
        # and actions that didn't less likely.

        log_like = tf.log(tf.reduce_sum(action.node * network.node))
        return -tf.reduce_sum(log_like * discounted_reward.node)


class Policy(Subgraph):
    def build_graph(self, network, loss):
        self.gradients = tf.gradients(loss.node, network.weights.node)
        return network.node

    def get_action(self, state):
        return Op(self.node, state=state)

    def compute_gradients(self, state, action, discounted_reward):
        return Op(
            self.gradients,
            state=state,
            action=action,
            discounted_reward=discounted_reward
        )


class ApplyGradients(Subgraph):
    def build_graph(self, optimizer, weights, gradients):
        self.apply_gradients_op = optimizer.node.apply_gradients(
            zip(gradients.node, weights.node)
        )

    def apply_gradients(self, gradients):
        return Op(self.apply_gradients_op, gradients=gradients)


class Adam(Subgraph):
    def build_graph(self, learning_rate=0.001):
        return tf.train.AdamOptimizer(learning_rate=learning_rate)


class Initialize(Subgraph):
    def build_graph(self):
        return tf.global_variables_initializer()

    def initialize(self):
        return Op(self.node)
