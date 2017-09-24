import numpy as np
import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import utils


class DA3CDiscreteLoss(subgraph.Subgraph):
    def build_graph(self, actor, critic, cfg):
        self.ph_action = graph.Placeholder(np.int32, shape=(None,))
        self.ph_value = graph.Placeholder(np.float32, shape=(None,))
        self.ph_discounted_reward = graph.Placeholder(np.float32, shape=(None,))

        td = self.ph_discounted_reward.node - self.ph_value.node
        if cfg.use_gae:
            self.ph_advantage = graph.Placeholder(np.float32, shape=(None,))
            td = self.ph_advantage.node

        action_one_hot = tf.one_hot(self.ph_action.node, actor.action_size)

        # avoid NaN
        log_pi = tf.log(tf.maximum(actor.node, 1e-20))

        # policy entropy
        self.entropy = -tf.reduce_sum(actor.node * log_pi)

        # policy loss
        self.policy_loss = -(tf.reduce_sum(tf.reduce_sum(log_pi * action_one_hot, axis=1) * td) +
                             self.entropy * cfg.entropy_beta)

        # value loss
        self.value_loss = tf.reduce_sum(tf.square(self.ph_discounted_reward.node - critic.node))

        # gradient of policy and value are summed up
        # (Learning rate for the Critic is sized by critic_scale parameter)
        return self.policy_loss + cfg.critic_scale * self.value_loss


class DQNLoss(subgraph.Subgraph):
    def build_graph(self, q_network, config):
        self.ph_reward = tf.placeholder(tf.float32, [None])
        self.ph_action = tf.placeholder(tf.int32, [None])
        self.ph_terminal = tf.placeholder(tf.int32, [None])
        self.ph_q_next_target = tf.placeholder(tf.float32, [None, config.output.action_size])
        self.ph_q_next = tf.placeholder(tf.float32, [None, config.output.action_size])

        action_one_hot = tf.one_hot(self.ph_action, config.output.action_size)
        q_action = tf.reduce_sum(tf.multiply(q_network.node, action_one_hot), axis=1)

        if config.double_dqn:
            q_max = tf.reduce_sum(self.ph_q_next_target * tf.one_hot(tf.argmax(self.ph_q_next, axis=1),
                                                                     config.output.action_size), axis=1)
        else:
            q_max = tf.reduce_max(self.ph_q_next_target, axis=1)

        y = self.ph_reward + tf.cast(1 - self.ph_terminal, tf.float32) * tf.scalar_mul(config.rewards_gamma,
                                                                                       q_max)

        return tf.losses.absolute_difference(q_action, y)


class DA3CNormContinuousLoss(subgraph.Subgraph):
    def build_graph(self, actor, critic, cfg):
        self.ph_action = graph.Placeholder(np.float32, shape=(None, actor.action_size))
        self.ph_value = graph.Placeholder(np.float32, shape=(None,))
        self.ph_discounted_reward = graph.Placeholder(np.float32, shape=(None,))

        td = self.ph_discounted_reward.node - self.ph_value.node
        if cfg.use_gae:
            self.ph_advantage = graph.Placeholder(np.float32, shape=(None,))
            td = self.ph_advantage.node

        mu, sigma2 = actor.node

        normal_dist = tf.contrib.distributions.Normal(mu, sigma2)
        log_prob = normal_dist.log_prob(self.ph_action.node)

        self.entropy = tf.reduce_sum(normal_dist.entropy())
        self.policy_loss = -(tf.reduce_sum(tf.reduce_sum(log_prob, axis=1) * td) +
                             cfg.entropy_beta * self.entropy)

        # Learning rate for the Critic is sized by critic_scale parameter
        self.value_loss = \
            cfg.critic_scale * tf.reduce_sum(tf.square(self.ph_discounted_reward.node - critic.node))


class DA3CExpContinuousLoss(subgraph.Subgraph):
    def build_graph(self, actor, critic, cfg):
        self.ph_action = graph.Placeholder(np.float32, shape=(None, actor.action_size))
        self.ph_value = graph.Placeholder(np.float32, shape=(None,))
        self.ph_discounted_reward = graph.Placeholder(np.float32, shape=(None,))

        td = self.ph_discounted_reward.node - self.ph_value.node
        if cfg.use_gae:
            self.ph_advantage = graph.Placeholder(np.float32, shape=(None,))
            td = self.ph_advantage.node

        mu, sigma2 = actor.node
        sigma2 += tf.constant(1e-6)

        log_std_dev = tf.log(sigma2)
        self.entropy = \
            tf.reduce_mean(log_std_dev + tf.constant(0.5 * np.log(2. * np.pi * np.e), tf.float32), axis=0)

        l2_dist = tf.square(self.ph_action.node - mu)
        sqr_std_dev = tf.constant(2.) * tf.square(sigma2) + tf.constant(1e-6)
        log_std_dev = tf.log(sigma2)
        log_prob = -l2_dist / sqr_std_dev - tf.constant(.5) * tf.log(tf.constant(2 * np.pi)) - log_std_dev

        self.policy_loss = -(tf.reduce_sum(tf.reduce_sum(log_prob, axis=1) * td) +
                             cfg.entropy_beta * self.entropy)

        # Learning rate for the Critic is sized by critic_scale parameter
        self.value_loss = \
            cfg.critic_scale * tf.reduce_sum(tf.square(self.ph_discounted_reward.node - critic.node))


class DA3CExtContinuousLoss(subgraph.Subgraph):
    def build_graph(self, actor, critic, cfg):
        self.ph_action = graph.Placeholder(np.float32, shape=(None, actor.action_size))
        self.ph_value = graph.Placeholder(np.float32, shape=(None,))
        self.ph_discounted_reward = graph.Placeholder(np.float32, shape=(None,))

        td = self.ph_discounted_reward.node - self.ph_value.node
        if cfg.use_gae:
            self.ph_advantage = graph.Placeholder(np.float32, shape=(None,))
            td = self.ph_advantage.node

        mu, sigma2 = actor.node
        sigma2 += tf.constant(1e-6)

        # policy entropy
        self.entropy = -tf.reduce_sum(0.5 * (tf.log(2. * np.pi * sigma2) + 1.), axis=1)

        # policy loss (calculation)
        b_size = tf.to_float(tf.size(self.ph_action.node) / actor.action_size)
        log_pi = tf.log(sigma2)
        x_prec = tf.exp(-log_pi)
        x_diff = tf.subtract(self.ph_action.node, mu)
        x_power = tf.square(x_diff) * x_prec * -0.5
        gaussian_nll = (tf.reduce_sum(log_pi, axis=1)
                        + b_size * tf.log(2. * np.pi)) / 2. - tf.reduce_sum(x_power, axis=1)

        self.policy_loss = -(tf.reduce_sum(gaussian_nll * td) + cfg.entropy_beta * self.entropy)

        # value loss
        # (Learning rate for the Critic is sized by critic_scale parameter)
        self.value_loss = \
            cfg.critic_scale * tf.reduce_sum(tf.square(self.ph_discounted_reward.node - critic.node))


def DA3CContinuousLoss(cfg):
    losses_list = {
        'Normal': DA3CNormContinuousLoss,
        'Expanded': DA3CExpContinuousLoss,
        'Extended': DA3CExtContinuousLoss
    }
    return losses_list[cfg.output.loss_type]


def DA3CLoss(actor, critic, cfg):
    Loss = DA3CContinuousLoss(cfg) if actor.continuous else DA3CDiscreteLoss
    return Loss(actor, critic, cfg)


class DDPGLoss(subgraph.Subgraph):
    def build_graph(self, critic_nn, cfg):
        loss = MeanSquaredLoss(critic_nn.critic)
        self.ph_predicted = loss.ph_predicted
        if cfg.l2:
            l2_loss = L2Loss(critic_nn.weights, cfg.l2_decay)
            loss = loss.node + l2_loss.node
        return loss


class MeanSquaredLoss(subgraph.Subgraph):
    def build_graph(self, y, size=1):
        self.ph_predicted = tf.placeholder(tf.float32, [None, size])
        return tf.reduce_mean(tf.square(self.ph_predicted - y.node))


class L2Loss(subgraph.Subgraph):
    def build_graph(self, weights, l2_decay=0.01):
        flattened_weights = list(utils.Utils.flatten(weights.node))
        l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in flattened_weights])
        return l2_decay * l2_loss


class PGLoss(subgraph.Subgraph):
    def build_graph(self, action_size, network):
        self.ph_action = graph.Placeholder(np.int32, (None,))
        self.ph_discounted_reward = graph.Placeholder(np.float32, (None, 1))

        # making actions that gave good advantage (reward over time) more likely,
        # and actions that didn't less likely.

        log_like_op = tf.log(tf.reduce_sum(tf.one_hot(self.ph_action.node,
                                                      action_size) * network.node, axis=[1]))
        return -tf.reduce_sum(log_like_op * self.ph_discounted_reward.node)
