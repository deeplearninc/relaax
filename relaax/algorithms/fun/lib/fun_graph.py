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


class ManagerLoss(subgraph.Subgraph):
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
