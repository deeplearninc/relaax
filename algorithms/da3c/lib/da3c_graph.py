import numpy as np
import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import utils

from .. import da3c_config


class Weights(subgraph.Subgraph):
    def build_graph(self):
        self.conv1 = graph.Wb(np.float32, (8, 8, 4, 16)) # stride=4
        self.conv2 = graph.Wb(np.float32, (4, 4, 16, 32)) # stride=2

        self.fc = graph.Wb(np.float32, (2592, 256))

        # weight for policy output layer
        self.actor = graph.Wb(np.float32, (256, da3c_config.config.action_size))

        # weight for value output layer
        self.critic = graph.Wb(np.float32, (256, 1))

        return map(lambda sg: sg.node, [
            self.conv1,
            self.conv2,
            self.fc,
            self.actor,
            self.critic
        ])


class Network(subgraph.Subgraph):
    def build_graph(self, state, weights):
        conv1 = graph.Relu(graph.Convolution(state, weights.conv1, 4))
        conv2 = graph.Relu(graph.Convolution(conv1, weights.conv2, 2))

        conv2_flat = graph.Reshape(conv2, [-1, 2592])
        fc = graph.Relu(graph.ApplyWb(conv2_flat, weights.fc))

        self.actor = graph.Softmax(graph.ApplyWb(fc, weights.actor))

        self.critic = graph.Reshape(graph.ApplyWb(fc, weights.critic), [-1])


class LearningRate(subgraph.Subgraph):
    def build_graph(self, global_step):
        n_steps = np.int64(da3c_config.config.max_global_step)
        reminder = tf.subtract(n_steps, global_step.n.node)
        factor = tf.cast(reminder, tf.float64) / tf.cast(n_steps, tf.float64)
        learning_rate = tf.maximum(tf.cast(0, tf.float64), factor * da3c_config.config.initial_learning_rate)
        return learning_rate


class ApplyGradients(subgraph.Subgraph):
    def build_graph(self, weights, gradients, learning_rate):
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=learning_rate.node,
            decay=da3c_config.config.RMSProp.decay,
            momentum=0.0,
            epsilon=da3c_config.config.RMSProp.epsilon
        )
        return optimizer.apply_gradients(utils.Utils.izip(gradients.node, weights.node))


class Loss(subgraph.Subgraph):
    def build_graph(self, state, action, value, discounted_reward, weights, actor, critic):
        action_one_hot = tf.one_hot(action.node, da3c_config.config.action_size)

        # avoid NaN with getting the maximum with small value
        log_pi = tf.log(tf.maximum(actor.node, 1e-20))

        # policy entropy
        entropy = -tf.reduce_sum(actor.node * log_pi, axis=1)

        # policy loss (output)  (Adding minus, because the original paper's
        # objective function is for gradient ascent, but we use gradient descent optimizer)
        policy_loss = -tf.reduce_sum(
            tf.reduce_sum(log_pi * action_one_hot, axis=1) * (discounted_reward.node - value.node) +
            entropy * da3c_config.config.entropy_beta
        )

        # value loss (output)
        # (Learning rate for Critic is half of Actor's, it's l2 without dividing by 0.5)
        value_loss = tf.reduce_sum(tf.square(discounted_reward.node - critic.node))

        # gradient of policy and value are summed up
        return policy_loss + value_loss
