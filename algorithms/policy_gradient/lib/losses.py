import tensorflow as tf


class SimpleLoss(object):
    @classmethod
    def assemble(cls, action, policy, discounted_reward):
        # making actions that gave good advantage (reward over time) more likely,
        # and actions that didn't less likely.

        print type(policy)
        log_like = tf.log(tf.reduce_sum(action * policy))
        return -tf.reduce_mean(log_like * discounted_reward)
