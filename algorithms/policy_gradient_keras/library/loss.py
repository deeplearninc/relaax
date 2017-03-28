from . import *


class SimpleLoss(object):
    def __init__(self, output):
        self.act = tf.placeholder(tf.float32, list(output.output_shape), name="taken_action")
        self.adv = tf.placeholder(tf.float32, [None], name="discounted_reward")

        log_like = tf.log(tf.reduce_sum(tf.multiply(self.act, output)))

        self.eval = -tf.reduce_mean(log_like * self.adv)
