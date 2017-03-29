import logging
import tensorflow as tf

from relaax.common.algorithms.decorators import define_scope, define_input
from pg_config import config

from lib.loss import Loss
from lib.weights import Weights
from lib.fc_network import FullyConnected
from lib.utils import assemble_and_show_graphs

log = logging.getLogger("policy_gradient")


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(object):

    def __init__(self):
        self.assemble()

    def assemble(self):
        # Build TF graph
        self.weights
        self.gradients
        self.apply_gradients

    @define_scope(initializer=tf.contrib.layers.xavier_initializer())
    def weights(self):
        return Weights.assemble(
            config.state_size, config.action_size, config.hidden_layers)

    @define_scope
    def gradients(self):
        # placeholders to apply gradients to shared parameters
        return [tf.placeholder(v.dtype, v.get_shape()) for v in self.weights]

    @define_scope
    def apply_gradients(self):
        # apply gradients to weights
        optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
        return optimizer.apply_gradients(zip(self.gradients, self.weights))


# Policy run by Agent(s)
class PolicyModel(Weights):

    def __init__(self):
        self.assemble()

    def assemble(self):
        # Build TF graph
        self.weights
        self.state
        self.action
        self.discounted_reward
        self.shared_weights
        self.policy
        self.partial_gradients
        self.assign_weights

    @define_input
    def state(self):
        return tf.placeholder(tf.float32, [None, config.state_size])

    @define_input
    def action(self):
        return tf.placeholder(tf.float32, [None, config.action_size])

    @define_input
    def discounted_reward(self):
        return tf.placeholder(tf.float32)

    @define_input
    def shared_weights(self):
        # placeholders to apply weights to shared parameters
        return [tf.placeholder(v.dtype, v.get_shape()) for v in self.weights]

    @define_scope
    def weights(self):
        return Weights.assemble(
            config.state_size, config.action_size, config.hidden_layers)

    @define_scope
    def policy(self):
        return FullyConnected.assemble_from_weights(self.state, self.weights)

    @define_scope
    def loss(self):
        return Loss.assemble(
            t_action=self.action, t_policy=self.policy, t_discounted_reward=self.discounted_reward)

    @define_scope
    def partial_gradients(self):
        return tf.gradients(self.loss, self.weights)

    @define_scope
    def assign_weights(self):
        return tf.group(*[
            tf.assign(v, p) for v, p in zip(self.weights, self.shared_weights)])


if __name__ == '__main__':
    assemble_and_show_graphs(
        parameter_server=SharedParameters, agent=PolicyModel)
