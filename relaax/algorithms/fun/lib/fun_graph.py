from __future__ import division
import numpy as np
import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph

from ..fun_config import config as cfg


class LearningRate(subgraph.Subgraph):
    def build_graph(self, global_step):
        n_steps = np.int64(cfg.anneal_step_limit)
        reminder = tf.subtract(n_steps, global_step.n.node)
        factor = tf.cast(reminder, tf.float64) / tf.cast(n_steps, tf.float64)
        factor = tf.maximum(factor, tf.cast(0, tf.float64))
        half_of_initial_learning_rate = tf.constant(cfg.initial_learning_rate / 2)
        learning_rate = half_of_initial_learning_rate * (factor + tf.constant(1.0))
        return learning_rate


class CosineLoss(subgraph.Subgraph):
    def build_graph(self, goal, critic):
        self.ph_stc_diff_st =\
            graph.Placeholder(np.float32, shape=(None, cfg.d), name="ph_stc_diff_st")
        s_diff_normalized = tf.nn.l2_normalize(self.ph_stc_diff_st.node, dim=1)

        cosine_similarity = tf.matmul(s_diff_normalized, goal.node, transpose_b=True)
        cosine_similarity = tf.diag_part(cosine_similarity)

        # manager's advantage (R-V): R = ri + cfg.wGAMMA * R; AdvM = R - ViM
        self.ph_discounted_reward =\
            graph.Placeholder(np.float32, shape=(None,), name="ph_m_discounted_reward")
        advantage = self.ph_discounted_reward.node - critic.node

        manager_loss = tf.reduce_sum(advantage * cosine_similarity)
        return manager_loss


class A3CLoss(subgraph.Subgraph):
    def build_graph(self, actor, critic, entropy=True):
        self.ph_action = graph.Placeholder(np.int32, shape=(None,), name="a")
        self.ph_value = graph.Placeholder(np.float32, shape=(None,), name="v")
        self.ph_discounted_reward = graph.Placeholder(np.float32, shape=(None,), name="r")

        action_one_hot = tf.one_hot(self.ph_action.node, cfg.action_size)

        # avoid NaN with getting the maximum with small value
        log_pi = tf.log(tf.maximum(actor.node, 1e-20))

        # policy entropy
        if entropy:
            entropy = -tf.reduce_sum(actor.node * log_pi, axis=1)
            # policy loss (output)  (Adding minus, because the original paper's
            # objective function is for gradient ascent, but we use gradient descent optimizer)
            policy_loss = -tf.reduce_sum(tf.reduce_sum(log_pi * action_one_hot, axis=1) *
                                         (self.ph_discounted_reward.node - self.ph_value.node) + entropy *
                                         cfg.entropy_beta)
        else:
            policy_loss = -tf.reduce_sum(tf.reduce_sum(log_pi * action_one_hot, axis=1) *
                                         (self.ph_discounted_reward.node - self.ph_value.node))

        # value loss (output)
        # (Learning rate for Critic is half of Actor's, it's l2 without dividing by 0.5)
        value_loss = tf.reduce_sum(tf.square(self.ph_discounted_reward.node - critic.node))

        # gradient of policy and value are summed up
        return policy_loss + value_loss
