import numpy as np
import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import utils

from .. import da3c_config


class Convolutions(subgraph.Subgraph):
    BORDER = {}
    ACTIVATION = {}

    def build_graph(self, x, convolutions):
        self.weight = []
        last = x
        for conv in convolutions:
            last = layer.Convolution(last, **self._edit(conv.clone())
            self.weight.append(last.weight)
        return last

    def _edit(self, conv):
        for key, mapping in [('border', self.BORDER),
                ('activation', self.ACTIVATION)]:
            if key in conv:
                conv[key] = mapping[conv[key]]
        return conv


        '''

    conv2:
      input: conv1
      type: convolution_2D
      n_filters: 32
      filter_size: [4, 4]
      stride: [2, 2]
      border: valid
      activation: relu


        conv1 = layer.Convolution2D(state, 16, 8, 8, subsample=(4, 4),
                border_mode='valid', activation=activation.Relu)
        conv2 = layer.Convolution2D(conv1, 32, 4, 4, subsample=(2, 2),
                border_mode='valid', activation=activation.Relu)
                '''
'''
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
'''

class Loss(subgraph.Subgraph):
    def build_graph(self, state, action, value, discounted_reward, actor, critic):
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


class LearningRate(subgraph.Subgraph):
    def build_graph(self, global_step):
        n_steps = np.int64(da3c_config.config.max_global_step)
        reminder = tf.subtract(n_steps, global_step.n.node)
        factor = tf.cast(reminder, tf.float64) / tf.cast(n_steps, tf.float64)
        learning_rate = tf.maximum(tf.cast(0, tf.float64), factor * da3c_config.config.initial_learning_rate)
        return learning_rate
