import tensorflow as tf

from relaax.common.algorithms.subgraph import Subgraph


class Loss(Subgraph):
    def build(self, action, policy, discounted_reward):
        # making actions that gave good advantage (reward over time) more likely,
        # and actions that didn't less likely.

        log_like = tf.log(tf.reduce_sum(action.op * policy.op))
        return -tf.reduce_mean(log_like * discounted_reward.op)
