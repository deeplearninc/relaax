import tensorflow as tf


class Loss(object):

    @classmethod
    def assemble(cls, t_action, t_policy, t_discounted_reward):
        # making actions that gave good
        # advantage (reward over time) more
        # likely, and actions that didn't less likely.
        log_like = tf.log(tf.reduce_sum(tf.multiply(t_action, t_policy)))
        return -tf.reduce_mean(log_like * t_discounted_reward)
