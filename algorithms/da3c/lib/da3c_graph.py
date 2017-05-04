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

    def get(self):
        return subgraph.Subgraph.Op(self.node)


class ApplyGradients(subgraph.Subgraph):
    def build_graph(self, weights, gradients, global_step):
        n_steps = np.int64(da3c_config.config.max_global_step)
        reminder = tf.subtract(n_steps, global_step.counter)
        factor = tf.cast(reminder, tf.float64) / tf.cast(n_steps, tf.float64)
        learning_rate = tf.maximum(tf.cast(0, tf.float64), factor * da3c_config.config.initial_learning_rate)

        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=learning_rate,
            decay=da3c_config.config.RMSProp.decay,
            momentum=0.0,
            epsilon=da3c_config.config.RMSProp.epsilon
        )
        return tf.group(
            optimizer.apply_gradients(utils.Utils.izip(gradients.node, weights.node)),
            global_step.increment
        )

    def apply_gradients(self, gradients, n_steps):
        return subgraph.Subgraph.Op(self.node, gradients=gradients, n_steps=n_steps)


class Network(subgraph.Subgraph):
    def build_graph(self, state, weights):
        conv1 = Relu(Convolution(state, weights.conv1, 4))
        conv2 = Relu(Convolution(conv1, weights.conv2, 2))

        conv2_flat = Reshape(conv2, [-1, 2592])
        fc = Relu(graph.ApplyWb(conv2_flat, weights.fc))

        self.actor = Softmax(graph.ApplyWb(fc, weights.actor))

        self.critic = Reshape(graph.ApplyWb(fc, weights.critic), [-1])

    def get_action_and_value(self, state):
        return subgraph.Subgraph.Op([self.actor.node, self.critic.node], state=state)


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
        loss = policy_loss + value_loss

        self.gradients = utils.Utils.reconstruct(
            tf.gradients(loss, list(utils.Utils.flatten(weights.node))),
            weights.node
        )

    def compute_gradients(self, state, action, value, discounted_reward):
        return subgraph.Subgraph.Op(
            self.gradients,
            state=state,
            action=action,
            value=value,
            discounted_reward=discounted_reward
        )


class LearningRate(subgraph.Subgraph):
    def build_graph(self, n_steps):
        factor = (self._config.max_global_step - global_time_step) / self._config.max_global_step
        learning_rate = self._config.INITIAL_LEARNING_RATE * factor
        if learning_rate < 0.0:
            learning_rate = 0.0
        return learning_rate
