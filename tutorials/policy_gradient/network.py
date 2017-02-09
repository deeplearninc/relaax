import tensorflow as tf


# Simple 2-layer fully-connected Policy Neural Network
class _PolicyNN(object):
    def __init__(self):
        self.global_t = tf.Variable(0, tf.int64)
        self.increment_global_t = tf.assign_add(self.global_t, 1)

    def compute_loss(self, config):
        return self

    def compute_gradients(self, config):
        return self

    def apply_gradients(self, config):
        return self


# Shared is used for global-NN and holds only weights on which applies computed gradients
class _PolicyNNShared(_PolicyNN):
    def __init__(self, config):
        super(_PolicyNNShared, self).__init__()

        self.W1 = tf.get_variable('W1', shape=(config.layer_size, config.state_size),
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.W2 = tf.get_variable('W2', shape=(config.layer_size,),
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.values = [
            self.W1, self.W2
        ]

        self._placeholders = [tf.placeholder(v.dtype, v.get_shape()) for v in self.values]
        self._assign_values = tf.group(*[
            tf.assign(v, p) for v, p in zip(self.values, self._placeholders)
            ])

        self.gradients = [tf.placeholder(v.dtype, v.get_shape()) for v in self.values]

    def assign_values(self, session, values):
        session.run(self._assign_values, feed_dict={
            p: v for p, v in zip(self._placeholders, values)
        })

    def get_vars(self):
        return self.values


class _PolicyNNFull(_PolicyNNShared):

    def __init__(self, config):
        super(_PolicyNNFull, self).__init__(config)

        # state (input)
        self.s = tf.placeholder(tf.float32, [None] + config.state_size)

        hidden_fc = tf.nn.relu(tf.matmul(self.W1, self.s))

        # policy (output)
        self.pi = tf.nn.sigmoid(tf.matmul(self.W1, hidden_fc))

    def run_policy(self, sess, s_t):
        pi_out = sess.run(self.pi, feed_dict={self.s: [s_t]})
        return pi_out[0]
