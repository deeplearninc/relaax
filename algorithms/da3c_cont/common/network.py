import tensorflow as tf
import numpy as np

from lstm import CustomBasicLSTMCell


def make_shared_network(config):
    if config.use_LSTM:
        network = _GameACLSTMNetworkShared(config)
    else:
        network = _GameACFFNetworkShared(config)
    return network.apply_gradients(config)


def make_full_network(config):
    if config.use_LSTM:
        network = _GameACLSTMNetwork(config)
    else:
        network = _GameACFFNetwork(config)
    return network.prepare_loss(config).compute_gradients(config)


# Actor-Critic Network Base Class
# (Policy network and Value network)
class _GameACNetwork(object):
    def __init__(self):
        self.global_t = tf.Variable(0, tf.int64)
        self.increment_global_t = tf.assign_add(self.global_t, 1)

    def prepare_loss(self, config):
        # taken action (input for policy)
        self.a = tf.placeholder("float", [None, config.action_size])

        # temporary difference (R-V) (input for policy)
        self.td = tf.placeholder("float", [None])

        # R (input for value)
        self.r = tf.placeholder("float", [None])

        if True:
            normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma2)
            loss = -tf.reduce_mean(tf.reduce_sum(normal_dist.log_prob(self.a), reduction_indices=1) * self.td)  # self.v
            loss -= 1e-1 * normal_dist.entropy()
            policy_loss = -tf.reduce_mean(loss)

            value_loss = tf.reduce_mean(tf.square(self.r - self.v))

            self.total_loss = policy_loss + value_loss
        else:
            # compute log and avoid NaN
            log_pi = tf.log(self.sigma2 + 1e-20)

            # policy entropy
            entropy = -tf.reduce_sum(0.5 * (tf.log(2. * np.pi * self.sigma2) + 1.), reduction_indices=1)

            # policy loss (output)
            b_size = tf.to_float(tf.size(self.a) / config.action_size)
            x_prec = tf.exp(-log_pi)
            x_diff = tf.sub(self.a, self.mu)
            x_power = tf.square(x_diff) * x_prec * -0.5
            gaussian_nll = (tf.reduce_sum(log_pi, reduction_indices=1)
                            + b_size * tf.log(2. * np.pi)) / 2. - tf.reduce_sum(x_power, reduction_indices=1)
            policy_loss = tf.mul(gaussian_nll, tf.stop_gradient(self.td)) + config.ENTROPY_BETA * entropy

            # value loss (output)
            # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
            value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

            # gradient of policy and value are summed up
            self.total_loss = policy_loss + value_loss

        return self

    def compute_gradients(self, config):
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate_input,
            decay=config.RMSP_ALPHA,
            momentum=0.0,
            epsilon=config.RMSP_EPSILON
        )
        grads_and_vars = optimizer.compute_gradients(self.total_loss, self.values)
        self.grads = [tf.clip_by_norm(grad, config.GRAD_NORM_CLIP)
                      for grad, _ in grads_and_vars]
        return self

    def apply_gradients(self, config):
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate_input,
            decay=config.RMSP_ALPHA,
            momentum=0.0,
            epsilon=config.RMSP_EPSILON
        )
        self.apply_gradients = optimizer.apply_gradients(zip(self.gradients, self.values))
        return self


class _GameACFFNetworkShared(_GameACNetwork):
    def __init__(self, config):
        super(_GameACFFNetworkShared, self).__init__()
        fc_size_1 = 300  # Size of the 1st fully connected layer
        fc_size_2 = 200  # Size of the 2nd fully connected layer
        fc_size_3 = 100  # Size of the 3rd fully connected layer
        self.input_size = np.prod(np.array(config.state_size)) * config.history_len

        # set of weights for fully connected layers (from input to heads)
        self.W_fc1 = _fc_weight_variable([self.input_size, fc_size_1])
        self.b_fc1 = _fc_bias_variable([fc_size_1], self.input_size)

        self.W_fc2 = _fc_weight_variable([fc_size_1, fc_size_2])
        self.b_fc2 = _fc_bias_variable([fc_size_2], fc_size_1)

        self.W_fc3 = _fc_weight_variable([fc_size_2, fc_size_3])
        self.b_fc3 = _fc_bias_variable([fc_size_3], fc_size_2)

        # weights for policy output layer
        self.W_fc4 = _fc_weight_variable([fc_size_3, config.action_size])
        self.b_fc4 = _fc_bias_variable([config.action_size], fc_size_3)

        self.W_fc5 = _fc_weight_variable([fc_size_3, config.action_size])
        self.b_fc5 = _fc_bias_variable([config.action_size], fc_size_3)

        # weights for value output layer
        self.W_fc6 = _fc_weight_variable([fc_size_3, 1])
        self.b_fc6 = _fc_bias_variable([1], fc_size_3)

        self.values = [
            self.W_fc1, self.b_fc1,
            self.W_fc2, self.b_fc2,
            self.W_fc3, self.b_fc3,
            self.W_fc4, self.b_fc4,
            self.W_fc5, self.b_fc5,
            self.W_fc6, self.b_fc6
        ]

        self._placeholders = [tf.placeholder(v.dtype, v.get_shape()) for v in self.values]
        self._assign_values = tf.group(*[
            tf.assign(v, p) for v, p in zip(self.values, self._placeholders)
            ])

        self.gradients = [tf.placeholder(v.dtype, v.get_shape()) for v in self.values]
        self.learning_rate_input = tf.placeholder(tf.float32)

    def assign_values(self, session, values):
        session.run(self._assign_values, feed_dict={
            p: v for p, v in zip(self._placeholders, values)
            })

    def get_vars(self):
        return self.values


# Actor-Critic FF Network
class _GameACFFNetwork(_GameACFFNetworkShared):
    def __init__(self, config):
        super(_GameACFFNetwork, self).__init__(config)

        # state (input)
        self.s = tf.placeholder("float", [None] + [self.input_size])

        h_fc1 = tf.nn.relu(tf.matmul(self.s, self.W_fc1) + self.b_fc1)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, self.W_fc3) + self.b_fc3)

        # policy (output)
        self.mu = tf.matmul(h_fc3, self.W_fc4) + self.b_fc4
        self.sigma2 = tf.nn.softplus(tf.matmul(h_fc3, self.W_fc5) + self.b_fc5) + 1e-5

        # value (output)
        v_ = tf.matmul(h_fc3, self.W_fc6) + self.b_fc6
        self.v = tf.reshape(v_, [-1])

    def run_policy_and_value(self, sess, s_t):
        mu_out, sig_out, v_out = sess.run([self.mu, self.sigma2, self.v], feed_dict={self.s: [s_t]})
        return mu_out[0], sig_out[0], v_out[0]

    def run_policy(self, sess, s_t):
        mu_out, sig_out = sess.run([self.mu, self.sigma2], feed_dict={self.s: [s_t]})
        return mu_out[0], sig_out[0]

    def run_value(self, sess, s_t):
        v_out = sess.run(self.v, feed_dict={self.s: [s_t]})
        return v_out[0]


class _GameACLSTMNetworkShared(_GameACNetwork):
    def __init__(self, config):
        super(_GameACLSTMNetworkShared, self).__init__()
        lstm_size = 128  # Size of the LSTM layer
        self.input_size = np.prod(np.array(config.state_size)) * config.history_len

        # set of weights for fully connected layers (from input to lstm)
        self.W_fc1 = _fc_weight_variable([self.input_size, lstm_size])
        self.b_fc1 = _fc_bias_variable([lstm_size], self.input_size)

        # lstm
        self.lstm = CustomBasicLSTMCell(lstm_size)

        # weights for policy output layer
        self.W_fc4 = _fc_weight_variable([lstm_size, config.action_size])
        self.b_fc4 = _fc_bias_variable([config.action_size], lstm_size)

        self.W_fc5 = _fc_weight_variable([lstm_size, config.action_size])
        self.b_fc5 = _fc_bias_variable([config.action_size], lstm_size)

        # weights for value output layer
        self.W_fc6 = _fc_weight_variable([lstm_size, 1])
        self.b_fc6 = _fc_bias_variable([1], lstm_size)

        # state (input)
        self.s = tf.placeholder("float", [None] + [self.input_size])

        h_fc1 = tf.nn.relu(tf.matmul(self.s, self.W_fc1) + self.b_fc1)

        # (batch_size, lstm_size) --> (1, batch_size, lstm_size)
        h_fc1_reshaped = tf.reshape(h_fc1, [1, -1, lstm_size])

        # place holder for LSTM unrolling time step size.
        self.step_size = tf.placeholder(tf.float32, [1])

        self.initial_lstm_state = tf.placeholder(tf.float32, [1, self.lstm.state_size])

        # Unrolling LSTM up to EPISODE_LEN time steps (=5 time steps)
        # When episode terminates unrolling time steps becomes less than LOCAL_TIME_STEP
        # Unrolling step size is applied via self.step_size placeholder
        # When forward propagating, step_size is 1
        # (time_major = False, so output shape is [batch_size, max_time, cell.output_size])
        self.lstm_outputs, self.lstm_state = tf.nn.dynamic_rnn(
            self.lstm,
            h_fc1_reshaped,
            initial_state=self.initial_lstm_state,
            sequence_length=self.step_size,
            time_major=False)

        self.values = [
            self.W_fc1, self.b_fc1,
            self.lstm.matrix, self.lstm.bias,
            self.W_fc4, self.b_fc4,
            self.W_fc5, self.b_fc5,
            self.W_fc6, self.b_fc6
        ]

        self._placeholders = [tf.placeholder(v.dtype, v.get_shape()) for v in self.values]
        self._assign_values = tf.group(*[
            tf.assign(v, p) for v, p in zip(self.values, self._placeholders)
            ])

        self.gradients = [tf.placeholder(v.dtype, v.get_shape()) for v in self.values]
        self.learning_rate_input = tf.placeholder(tf.float32)

    def assign_values(self, session, values):
        session.run(self._assign_values, feed_dict={
            p: v for p, v in zip(self._placeholders, values)
            })

    def get_vars(self):
        return self.values


# Actor-Critic LSTM Network
class _GameACLSTMNetwork(_GameACLSTMNetworkShared):
    def __init__(self, config):
        super(_GameACLSTMNetwork, self).__init__(config)

        # lstm_outputs: (1, batch_size, lstm_size) for back prop; (1, batch_size, lstm_size) for forward prop
        lstm_outputs = tf.reshape(self.lstm_outputs, [-1, self.lstm.state_size / 2])

        # policy (output)
        self.mu = tf.matmul(lstm_outputs, self.W_fc4) + self.b_fc4
        self.sigma2 = tf.nn.softplus(tf.matmul(lstm_outputs, self.W_fc5) + self.b_fc5) + 1e-5

        # value (output)
        v_ = tf.matmul(lstm_outputs, self.W_fc6) + self.b_fc6
        self.v = tf.reshape(v_, [-1])

        self.reset_state()

    def reset_state(self):
        self.lstm_state_out = np.zeros([1, self.lstm.state_size])

    def run_policy_and_value(self, sess, s_t):
        # This run_policy_and_value() is used when forward propagating, so the step size is 1.
        mu_out, sig_out, v_out, self.lstm_state_out = sess.run([self.mu, self.sigma2, self.v, self.lstm_state],
                                                               feed_dict={self.s: [s_t],
                                                                          self.initial_lstm_state: self.lstm_state_out,
                                                                          self.step_size: [1]})
        return mu_out[0], sig_out[0], v_out[0]

    def run_policy(self, sess, s_t):
        # This run_policy() is used for displaying the result with display tool.
        mu_out, sig_out, self.lstm_state_out = sess.run([self.mu, self.sigma2, self.lstm_state],
                                                        feed_dict={self.s: [s_t],
                                                                   self.initial_lstm_state: self.lstm_state_out,
                                                                   self.step_size: [1]})
        return mu_out[0], sig_out[0]

    def run_value(self, sess, s_t):
        # This run_value() is used for calculating V for bootstrapping at the
        # end of EPISODE_LEN time step sequence.
        # When next sequent starts, V will be calculated again with the same state using updated network weights,
        # so we don't update LSTM state here.
        prev_lstm_state_out = self.lstm_state_out
        v_out, _ = sess.run([self.v, self.lstm_state],
                            feed_dict={self.s: [s_t],
                                       self.initial_lstm_state: self.lstm_state_out,
                                       self.step_size: [1]})

        # roll back lstm state
        self.lstm_state_out = prev_lstm_state_out
        return v_out[0]


def assign_vars(dst_network, src_network):
    with tf.name_scope(None, "GameACNetwork", []) as name:
        sync_ops = []
        for src_var, dst_var in zip(src_network.get_vars(), dst_network.get_vars()):
            with tf.device(dst_var.device):
                sync_ops.append(tf.assign(dst_var, src_var))
        return tf.group(*sync_ops, name=name)


def _conv_weight_variable(shape):
    w = shape[0]
    h = shape[1]
    input_channels = shape[2]
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial)


def _conv_bias_variable(shape, w, h, input_channels):
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial)


# weight initialization
def _fc_weight_variable(shape):
    input_channels = shape[0]
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial)


def _fc_bias_variable(shape, input_channels):
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial)


def _conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="VALID")
