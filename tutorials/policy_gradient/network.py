import tensorflow as tf
import numpy as np


# Simple 2-layer fully-connected Policy Neural Network
class _GlobalPolicyNN(object):
    # This class is used for global-NN and holds only weights on which applies computed gradients
    def __init__(self, config):
        self.global_t = tf.Variable(0, tf.int64)
        self.increment_global_t = tf.assign_add(self.global_t, 1)

        self._GAMMA = config.GAMMA
        self._RMSP_DECAY = config.RMSP_DECAY
        self._RMSP_EPSILON = config.RMSP_EPSILON

        self.W1 = tf.get_variable('W1', shape=[config.layer_size, config.state_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.W2 = tf.get_variable('W2', shape=[config.layer_size, 1],
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.values = [
            self.W1, self.W2
        ]

        self._placeholders = [tf.placeholder(v.dtype, v.get_shape()) for v in self.values]
        self._assign_values = tf.group(*[
            tf.assign(v, p) for v, p in zip(self.values, self._placeholders)
            ])

        self.gradients = [tf.placeholder(v.dtype, v.get_shape()) for v in self.values]
        self.learning_rate = tf.placeholder(tf.float32)

    def assign_values(self, session, values):
        session.run(self._assign_values, feed_dict={
            p: v for p, v in zip(self._placeholders, values)
            })

    def get_vars(self):
        return self.values

    def apply_gradients(self):
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate,
            decay=self._RMSP_DECAY,
            epsilon=self._RMSP_EPSILON
        )
        self.apply_gradients = optimizer.apply_gradients(zip(self.gradients, self.values))
        return self


class _AgentPolicyNN(_GlobalPolicyNN):
    # This class additionally implements loss computation and gradients wrt this loss
    def __init__(self, config):
        super(_AgentPolicyNN, self).__init__(config)

        # state (input)
        self.s = tf.placeholder(tf.float32, [None] + config.state_size)

        hidden_fc = tf.nn.relu(tf.matmul(self.W1, self.s))

        # policy (output)
        self.pi = tf.nn.sigmoid(tf.matmul(self.W1, hidden_fc))

    def run_policy(self, sess, s_t):
        pi_out = sess.run(self.pi, feed_dict={self.s: [s_t]})
        return pi_out[0]

    def compute_gradients(self):
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate,
            decay=self._RMSP_DECAY,
            epsilon=self._RMSP_EPSILON
        )
        grads_and_vars = optimizer.compute_gradients(self.loss, self.values)
        self.grads = [grad for grad, _ in grads_and_vars]
        return self

    def compute_loss(self):
        self.loss = None
        return self

    def discount_rewards(self, r):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, r.size)):
            if r[t] != 0:
                running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * self._GAMMA + r[t]
            discounted_r[t] = running_add
        return discounted_r
