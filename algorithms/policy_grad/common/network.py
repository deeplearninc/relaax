import tensorflow as tf
import numpy as np


def make_full_network(config):
    network = AgentPolicyNN(config)
    return network.prepare_loss().compute_gradients()


def make_shared_network(config):
    network = GlobalPolicyNN(config)
    return network.apply_gradients()


# Simple 2-layer fully-connected Policy Neural Network
class GlobalPolicyNN(object):
    # This class is used for global-NN and holds only weights on which applies computed gradients
    def __init__(self, config):
        self.global_t = tf.Variable(0, tf.int64)
        self.increment_global_t = tf.assign_add(self.global_t, 1)

        self._input_size = np.prod(np.array(config.state_size))
        self._action_size = config.action_size

        if type(config.layers_size) not in [list, tuple]:
            config.layers_size = [config.layers_size]

        self.values = [tf.get_variable('W0', shape=[self._input_size, config.layers_size[0]],
                                       initializer=tf.contrib.layers.xavier_initializer())]
        idx = len(config.layers_size)
        for i in range(1, idx):
            self.values.append(tf.get_variable('W%d' % i, shape=[config.layers_size[i-1], config.layers_size[i]],
                                               initializer=tf.contrib.layers.xavier_initializer()))
        self.values.append(tf.get_variable('W%d' % idx, shape=[config.layers_size[-1], self._action_size],
                                           initializer=tf.contrib.layers.xavier_initializer()))

        self._placeholders = [tf.placeholder(v.dtype, v.get_shape()) for v in self.values]
        self._assign_values = tf.group(*[
            tf.assign(v, p) for v, p in zip(self.values, self._placeholders)
            ])

        self.gradients = [tf.placeholder(v.dtype, v.get_shape()) for v in self.values]
        self.learning_rate = config.learning_rate

    def assign_values(self, session, values):
        session.run(self._assign_values, feed_dict={
            p: v for p, v in zip(self._placeholders, values)
            })

    def get_vars(self):
        return self.values

    def apply_gradients(self):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate
        )
        self.apply_gradients = optimizer.apply_gradients(zip(self.gradients, self.values))
        return self


class AgentPolicyNN(GlobalPolicyNN):
    # This class additionally implements loss computation and gradients wrt this loss
    def __init__(self, config):
        super(AgentPolicyNN, self).__init__(config)

        # state (input)
        self.s = tf.placeholder(tf.float32, [None, self._input_size])

        structure = [tf.nn.relu(tf.matmul(self.s, self.values[0]))]
        for i in range(1, len(self.values) - 1):
            structure.append(tf.nn.relu(tf.matmul(structure[i-1], self.values[i])))

        # policy (output)
        self.pi = tf.nn.sigmoid(tf.matmul(structure[-1], self.values[-1]))

    def run_policy(self, sess, s_t):
        pi_out = sess.run(self.pi, feed_dict={self.s: [s_t]})
        return pi_out[0]

    def compute_gradients(self):
        self.grads = tf.gradients(self.loss, self.values)
        return self

    def prepare_loss(self):
        self.a = tf.placeholder(tf.float32, [None, self._action_size], name="taken_action")
        self.advantage = tf.placeholder(tf.float32, name="discounted_reward")

        # making actions that gave good advantage (reward over time) more likely,
        # and actions that didn't less likely.
        log_like = tf.log(tf.reduce_sum(tf.multiply(self.a, self.pi)))
        self.loss = -tf.reduce_mean(log_like * self.advantage)

        return self
