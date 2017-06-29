import numpy as np
import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph


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

        # avoid NaN with getting the maximum with small value
        log_pi = tf.log(tf.maximum(actor.node, 1e-20))

        # policy entropy
        entropy = -tf.reduce_sum(actor.node * log_pi, axis=1)

        # policy loss (output)  (Adding minus, because the original paper's
        # objective function is for gradient ascent, but we use gradient descent optimizer)
        policy_loss = -tf.reduce_sum(tf.reduce_sum(log_pi * action_one_hot, axis=1) * td
                                     + entropy * cfg.entropy_beta)

        # value loss (output)
        # (Learning rate for Critic is half of Actor's, it's l2 without dividing by 0.5)
        value_loss = tf.reduce_sum(tf.square(self.ph_discounted_reward.node - critic.node))

        # gradient of policy and value are summed up
        return policy_loss + value_loss


class DA3CNormContinuousLoss(subgraph.Subgraph):
    def build_graph(self, actor, critic, cfg):
        self.ph_action = graph.Placeholder(np.float32, shape=(None, actor.action_size))
        self.ph_value = graph.Placeholder(np.float32, shape=(None,))
        self.ph_discounted_reward = graph.Placeholder(np.float32, shape=(None,))

        td = self.ph_discounted_reward.node - self.ph_value.node
        mu, sigma2 = actor.node

        normal_dist = tf.contrib.distributions.Normal(mu, sigma2 + 1e-6)
        log_prob = normal_dist.log_prob(self.ph_action.node)

        entropy = cfg.entropy_beta * normal_dist.entropy()
        policy_loss = -tf.reduce_sum(tf.reduce_sum(log_prob + entropy, axis=1) * td)

        # value loss (output)
        # (Learning rate for Critic is half of Actor's, it's l2 without dividing by 0.5)
        value_loss = tf.reduce_sum(tf.square(self.ph_discounted_reward.node - critic.node))

        # gradient of policy and value are summed up
        return policy_loss + value_loss


class DA3CExpContinuousLoss(subgraph.Subgraph):
    def build_graph(self, actor, critic, cfg):
        self.ph_action = graph.Placeholder(np.float32, shape=(None, actor.action_size))
        self.ph_value = graph.Placeholder(np.float32, shape=(None,))
        self.ph_discounted_reward = graph.Placeholder(np.float32, shape=(None,))

        td = self.ph_discounted_reward.node - self.ph_value.node
        mu, sigma2 = actor.node
        sigma2 += tf.constant(1e-6)

        log_std_dev = tf.log(sigma2)
        entropy = tf.reduce_mean(log_std_dev + tf.constant(0.5 * np.log(2. * np.pi * np.e), tf.float32), axis=0)

        l2_dist = tf.square(self.ph_action.node - mu)
        sqr_std_dev = tf.constant(2.) * tf.square(sigma2) + tf.constant(1e-6)
        log_std_dev = tf.log(sigma2)
        log_prob = -l2_dist / sqr_std_dev \
                   - tf.constant(.5) * tf.log(tf.constant(2 * np.pi)) - log_std_dev

        policy_loss = -tf.reduce_sum(tf.reduce_sum(log_prob + cfg.entropy_beta * entropy, axis=1) * td)
        value_loss = tf.reduce_sum(tf.square(self.ph_discounted_reward.node - critic.node))

        # gradient of policy and value are summed up
        return policy_loss + value_loss


class DA3CExtContinuousLoss(subgraph.Subgraph):
    def build_graph(self, actor, critic, cfg):
        self.ph_action = graph.Placeholder(np.float32, shape=(None, actor.action_size))
        self.ph_value = graph.Placeholder(np.float32, shape=(None,))
        self.ph_discounted_reward = graph.Placeholder(np.float32, shape=(None,))

        mu, sigma2 = actor.node
        sigma2 += tf.constant(1e-6)
        td = self.ph_discounted_reward.node - self.ph_value.node

        # policy entropy
        entropy = -tf.reduce_sum(0.5 * (tf.log(2. * np.pi * sigma2) + 1.), axis=1)

        # policy loss (output)
        b_size = tf.to_float(tf.size(self.ph_action.node) / actor.action_size)
        log_pi = tf.log(sigma2)
        x_prec = tf.exp(-log_pi)
        x_diff = tf.subtract(self.ph_action.node, mu)
        x_power = tf.square(x_diff) * x_prec * -0.5
        gaussian_nll = (tf.reduce_sum(log_pi, axis=1)
                        + b_size * tf.log(2. * np.pi)) / 2. - tf.reduce_sum(x_power, axis=1)

        policy_loss = -tf.reduce_sum(
            tf.multiply(gaussian_nll, tf.stop_gradient(td)) + cfg.entropy_beta * entropy
        )

        # value loss (output)
        # (Learning rate for Critic is half of Actor's, it's l2 without dividing by 0.5)
        value_loss = tf.reduce_sum(tf.square(self.ph_discounted_reward.node - critic.node))

        # gradient of policy and value are summed up
        return policy_loss + value_loss


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


class PGLoss(subgraph.Subgraph):
    def build_graph(self, action_size, network):
        self.ph_action = graph.Placeholder(np.int32, (None,))
        self.ph_discounted_reward = graph.Placeholder(np.float32, (None, 1))

        # making actions that gave good advantage (reward over time) more likely,
        # and actions that didn't less likely.

        log_like_op = tf.log(tf.reduce_sum(tf.one_hot(self.ph_action.node,
                                                      action_size) * network.node, axis=[1]))
        return -tf.reduce_sum(log_like_op * self.ph_discounted_reward.node)


class ICMLoss(subgraph.Subgraph):  # alpha=0.1 | beta=0.2
    def build_graph(self, policy_actor, icm_nn, alpha, beta):
        self.ph_action = graph.Placeholder(np.int32, (None,), name='action_from_policy')
        self.ph_discounted_reward = graph.Placeholder(np.float32, (None, 1), name='dr')

        action_one_hot = tf.one_hot(self.ph_action.node, policy_actor.action_size)
        action_log_prob = tf.log(tf.maximum(policy_actor.node, 1e-20))

        log_like = tf.reduce_sum(action_log_prob * action_one_hot, axis=1)
        policy_loss = -tf.reduce_sum(log_like * self.ph_discounted_reward.node) * alpha

        icm_action = tf.maximum(icm_nn.inv_out.node, 1e-20)
        max_like_sum = tf.reduce_sum(icm_action * action_one_hot)
        inv_loss = (1 - beta) * max_like_sum

        print('icm_nn.discrepancy', icm_nn.discrepancy.get_shape())
        fwd_loss = tf.reduce_sum(icm_nn.discrepancy) * beta

        return policy_loss + inv_loss + fwd_loss
