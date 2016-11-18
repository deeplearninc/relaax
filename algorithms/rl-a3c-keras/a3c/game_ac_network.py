import tensorflow as tf
import numpy as np
from lstm import CustomBasicLSTMCell

from keras.models import Model
from keras.layers import Input, Convolution2D, Reshape, Dense, \
    Merge, merge, Flatten, LSTM, Lambda, TimeDistributed, RepeatVector
from keras.initializations import normal
from keras import backend as K
from keras.models import model_from_json
import os


# Actor-Critic Network Base Class
# (Policy network and Value network)
class GameACNetwork(object):
    def __init__(self,
                 action_size, session,
                 device="/cpu:0"):
        self._device = device
        self._action_size = action_size
        K.set_session(session)

    def prepare_loss(self, entropy_beta):
        with tf.device(self._device):
            # taken action (input for policy)
            # self.a = K.placeholder(shape=(None, self._action_size), dtype="float")
            self.a = tf.placeholder("float", [None, self._action_size])

            # temporary difference (R-V) (input for policy)
            # self.td = K.placeholder(shape=None, dtype="float")
            self.td = tf.placeholder("float", [None])

            # avoid NaN with clipping when value in pi becomes zero
            # log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0))  # last ground truth
            log_pi = tf.log(tf.maximum(self.pi, 1e-20))

            # log_pi = tf.log(tf.select(tf.greater(self.pi, 1e-20), self.pi, tf.clip_by_value(self.pi, 1e-20, 1.0)))
            # entropy = -tf.reduce_sum(self.pi * tf.log(self.pi), reduction_indices=1)

            # policy entropy
            entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)

            # policy loss (output)
            # Adding minus, because the original paper's objective function is for gradient ascent,
            # but we use gradient descent optimizer
            policy_loss = - tf.reduce_sum(
                tf.reduce_sum(tf.mul(log_pi, self.a), reduction_indices=1) * self.td + entropy * entropy_beta)

            # R (input for value)
            # self.r = K.placeholder(shape=None, dtype="float")
            self.r = tf.placeholder("float", [None])    # [None]

            # value loss (output)
            # (Learning rate for Critic is half of Actor's, so multiply by 0.5)
            value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v)

            # gradient of policy and value are summed up
            self.total_loss = policy_loss + value_loss
            # return self.total_loss

    def run_policy_and_value(self, sess, s_t):
        raise NotImplementedError()

    def run_policy(self, sess, s_t):
        raise NotImplementedError()

    def run_value(self, sess, s_t):
        raise NotImplementedError()

    def get_vars(self):
        raise NotImplementedError()

    def sync_from(self, src_netowrk, name=None):
        src_vars = src_netowrk.get_vars()
        # self.net.set_weights(src_vars)
        dst_vars = self.get_vars()

        sync_ops = []

        with tf.device(self._device):
            with tf.op_scope([], name, "GameACNetwork") as name:
                for (src_var, dst_var) in zip(src_vars, dst_vars):
                    sync_op = tf.assign(dst_var, src_var)
                    sync_ops.append(sync_op)

                return tf.group(*sync_ops, name=name)

    def save(self, n_iter, chk_dir):
        if not os.path.exists(chk_dir):
            os.makedirs(chk_dir)
        self.net.save_weights(chk_dir + "/net--" + str(n_iter) + ".h5")

    def restore(self, chk_dir):
        n_iter = 0
        # just for tests
        if os.path.exists(chk_dir):
            file_names = [fn for fn in os.listdir(chk_dir) if fn.endswith('.h5')]
            if len(file_names) > 0:
                tokens = file_names[-1].split("--")
                n_iter = int(tokens[1].split(".")[0])
                fname = "/net--" + str(n_iter) + ".h5"
                self.net.load_weights(chk_dir + fname)
                print("checkpoint loaded:", fname)
                print(">>> global step set: ", n_iter)
            else:
                print("Could not find old checkpoint")
        else:
            print("Could not find old checkpoint")
        return n_iter

# Actor-Critic FF Network
class GameACFFNetwork(GameACNetwork):
    def __init__(self,
                 action_size, session,
                 device="/cpu:0"):
        GameACNetwork.__init__(self, action_size, session, device)

        with tf.device(self._device):
            if not os.path.isfile('model.json'):
                # state (input)
                # self.s = K.placeholder(shape=(None, 84, 84, 4), dtype="float")
                self.s = Input(shape=(84, 84, 4), dtype="float")    # [..]  tf.float32  K.placeholder  S

                # h_conv1 = tf.nn.relu(self._conv2d(self.s, self.W_conv1, 4) + self.b_conv1)
                h_conv1 = Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='valid',
                                        activation='relu', dim_ordering='tf')(self.s)   # S
                # h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)
                h_conv2 = Convolution2D(32, 4, 4, subsample=(2, 2), border_mode='valid',
                                        activation='relu', dim_ordering='tf')(h_conv1)

                # h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
                h_conv2_flat = Flatten()(h_conv2)
                # h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)
                h_fc1 = Dense(256, activation='relu')(h_conv2_flat)

                # policy (output)
                # self.pi = tf.nn.softmax(tf.matmul(h_fc1, self.W_fc2) + self.b_fc2)
                self.pi = Dense(action_size, activation='softmax')(h_fc1)
                # value (output)
                # v_ = tf.matmul(h_fc1, self.W_fc3) + self.b_fc3
                v_ = Dense(1)(h_fc1)    # v_
                # self.v = Reshape((1, ))(v_)   # need reverse shape
                self.v = tf.reshape(v_, [-1])

                # out = merge([self.pi, self.v], mode='concat')
                self.net = Model(input=self.s, output=[self.pi, v_])

                model_json = self.net.to_json()
                with open("model.json", "w") as json_file:
                    json_file.write(model_json)
            else:
                # load json and create model
                json_file = open('model.json', 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                self.net = model_from_json(loaded_model_json)

                self.s = self.net.input
                self.pi = self.net.output[0]
                v_ = self.net.output[1]
                self.v = tf.reshape(v_, [-1])

    def run_policy_and_value(self, sess, s_t):
        pi_out, v_out = sess.run([self.pi, self.v], feed_dict={self.s: [s_t]})
        return (pi_out[0], v_out[0])

    def run_policy(self, sess, s_t):
        pi_out = sess.run(self.pi, feed_dict={self.s: [s_t]})
        return pi_out[0]

    def run_value(self, sess, s_t):
        v_out = sess.run(self.v, feed_dict={self.s: [s_t]})
        return v_out[0]

    def get_vars(self):
        return self.net.trainable_weights


# Actor-Critic LSTM Network
class GameACLSTMNetwork(GameACNetwork):
    def __init__(self,
                 action_size, session,
                 thread_index,  # -1 for global
                 device="/cpu:0"):
        GameACNetwork.__init__(self, action_size, session, device)

        with tf.device(self._device):
            if not os.path.isfile('model-lstm.json'):
                # state (input)
                # self.s = tf.placeholder("float", [None, 84, 84, 4])
                self.s = Input(shape=(84, 84, 4), dtype="float")
                # Lambda(lambda x: tf.transpose(self.s, [2, 0, 1]))

                # h_conv1 = tf.nn.relu(self._conv2d(self.s, self.W_conv1, 4) + self.b_conv1)
                h_conv1 = Convolution2D(16, 8, 8, subsample=(4, 4), border_mode='valid',
                                        activation='relu', dim_ordering='tf')(self.s)
                # h_conv2 = tf.nn.relu(self._conv2d(h_conv1, self.W_conv2, 2) + self.b_conv2)
                h_conv2 = Convolution2D(32, 4, 4, subsample=(2, 2), border_mode='valid',
                                        activation='relu', dim_ordering='tf')(h_conv1)

                # h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
                h_conv2_flat = Flatten()(h_conv2)
                # h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, self.W_fc1) + self.b_fc1)
                h_fc1 = Dense(256, activation='relu')(h_conv2_flat)
                # h_fc1 shape = (5,256)

                # h_fc1_reshaped = tf.reshape(h_fc1, [1, -1, 256])
                h_fc1_reshaped = RepeatVector(1)(h_fc1)
                # h_fc1_reshaped shape = (1,5,256)

                lstm_outputs = LSTM(256)(h_fc1_reshaped)

                # policy (output)
                # self.pi = tf.nn.softmax(tf.matmul(lstm_outputs, self.W_fc2) + self.b_fc2)
                self.pi = Dense(action_size, activation='softmax')(lstm_outputs)

                # value (output)
                # v_ = tf.matmul(lstm_outputs, self.W_fc3) + self.b_fc3
                v_ = Dense(1)(h_fc1)
                self.v = tf.reshape(v_, [-1])

                self.net = Model(input=self.s, output=[self.pi, v_])
                # self.reset_state()

                model_json = self.net.to_json()
                with open("model-lstm.json", "w") as json_file:
                    json_file.write(model_json)
            else:
                # load json and create model
                json_file = open('model-lstm.json', 'r')
                loaded_model_json = json_file.read()
                json_file.close()
                self.net = model_from_json(loaded_model_json)

                self.s = self.net.input
                self.pi = self.net.output[0]
                v_ = self.net.output[1]
                self.v = tf.reshape(v_, [-1])

    def reset_state(self):
        self.lstm_state_out = np.zeros([1, self.lstm.state_size])

    def run_policy_and_value(self, sess, s_t):
        # This run_policy_and_value() is used when forward propagating.
        # so the step size is 1.
        '''
        pi_out, v_out, self.lstm_state_out = sess.run([self.pi, self.v, self.lstm_state],
                                                      feed_dict={self.s: [s_t],
                                                                 self.initial_lstm_state: self.lstm_state_out,
                                                                 self.step_size: [1]})'''
        pi_out, v_out = sess.run([self.pi, self.v], feed_dict={self.s: [s_t]})
        # pi_out: (1,3), v_out: (1)
        return (pi_out[0], v_out[0])

    def run_policy(self, sess, s_t):
        # This run_policy() is used for displaying the result with display tool.
        '''
        pi_out, self.lstm_state_out = sess.run([self.pi, self.lstm_state],
                                               feed_dict={self.s: [s_t],
                                                          self.initial_lstm_state: self.lstm_state_out,
                                                          self.step_size: [1]})'''
        pi_out = sess.run(self.pi, feed_dict={self.s: [s_t]})
        return pi_out[0]

    def run_value(self, sess, s_t):
        # This run_value() is used for calculating V for bootstrapping at the
        # end of LOCAL_T_MAX time step sequence.
        # When next sequent starts, V will be calculated again with the same state using updated network weights,
        # so we don't update LSTM state here.
        '''
        prev_lstm_state_out = self.lstm_state_out
        v_out, _ = sess.run([self.v, self.lstm_state],
                            feed_dict={self.s: [s_t],
                                       self.initial_lstm_state: self.lstm_state_out,
                                       self.step_size: [1]})

        # roll back lstm state
        self.lstm_state_out = prev_lstm_state_out'''
        v_out = sess.run(self.v, feed_dict={self.s: [s_t]})
        return v_out[0]

    def get_vars(self):
        return self.net.trainable_weights
