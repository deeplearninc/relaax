import tensorflow as tf
import math
from params import *


class CriticNet:
    """ Critic Q value model of the DDPG algorithm """

    def __init__(self, num_states, num_actions):
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.InteractiveSession()

            # critic_q_model parameters:
            self.W1_c, self.B1_c, self.W2_c, self.W2_action_c, self.B2_c, self.W3_c, self.B3_c, \
            self.critic_q_model, self.critic_state_in, self.critic_action_in = self.create_critic_net(num_states,
                                                                                                      num_actions)

            # create target_q_model:
            self.t_W1_c, self.t_B1_c, self.t_W2_c, self.t_W2_action_c, self.t_B2_c, self.t_W3_c, self.t_B3_c, \
            self.t_critic_q_model, self.t_critic_state_in, self.t_critic_action_in = self.create_critic_net(num_states,
                                                                                                            num_actions)

            self.q_value_in = tf.placeholder("float", [None, 1])  # supervisor
            self.l2_regularizer_loss = 0.0001 * tf.reduce_sum(tf.pow(self.W2_c, 2)) + 0.0001 * tf.reduce_sum(
                tf.pow(self.B2_c, 2))
            self.cost = tf.pow(self.critic_q_model - self.q_value_in,
                               2) / BATCH_SIZE + self.l2_regularizer_loss  # /tf.to_float(tf.shape(self.q_value_in)[0])
            self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost)

            # action gradient to be used in actor network:
            # self.action_gradients=tf.gradients(self.critic_q_model,self.critic_action_in)
            # from simple actor net:
            self.act_grad_v = tf.gradients(self.critic_q_model, self.critic_action_in)
            self.action_gradients = [
                self.act_grad_v[0] / tf.to_float(tf.shape(self.act_grad_v[0])[0])]  # this is just divided by batch size
            # from simple actor net:
            self.check_fl = self.action_gradients

            # temporary
            self.TAU = tf.constant(TAU)         # TAU
            self.TAU_ = tf.constant(1 - TAU)    # 1 - TAU

            # initialize all tensor variable parameters:
            self.sess.run(tf.initialize_all_variables())

            # To make sure critic and target have same parameters copy the parameters:
            # copy target parameters
            self.sess.run([
                self.t_W1_c.assign(self.W1_c),
                self.t_B1_c.assign(self.B1_c),
                self.t_W2_c.assign(self.W2_c),
                self.t_W2_action_c.assign(self.W2_action_c),
                self.t_B2_c.assign(self.B2_c),
                self.t_W3_c.assign(self.W3_c),
                self.t_B3_c.assign(self.B3_c)
            ])

    @staticmethod
    def create_critic_net(num_states=4, num_actions=1):
        critic_state_in = tf.placeholder("float", [None, num_states])
        critic_action_in = tf.placeholder("float", [None, num_actions])

        W1_c = tf.Variable(
            tf.random_uniform([num_states, N_HIDDEN_1], -1 / math.sqrt(num_states), 1 / math.sqrt(num_states)))
        B1_c = tf.Variable(tf.random_uniform([N_HIDDEN_1], -1 / math.sqrt(num_states), 1 / math.sqrt(num_states)))
        W2_c = tf.Variable(tf.random_uniform([N_HIDDEN_1, N_HIDDEN_2], -1 / math.sqrt(N_HIDDEN_1 + num_actions),
                                             1 / math.sqrt(N_HIDDEN_1 + num_actions)))
        W2_action_c = tf.Variable(tf.random_uniform([num_actions, N_HIDDEN_2], -1 / math.sqrt(N_HIDDEN_1 + num_actions),
                                                    1 / math.sqrt(N_HIDDEN_1 + num_actions)))
        B2_c = tf.Variable(tf.random_uniform([N_HIDDEN_2], -1 / math.sqrt(N_HIDDEN_1 + num_actions),
                                             1 / math.sqrt(N_HIDDEN_1 + num_actions)))
        W3_c = tf.Variable(tf.random_uniform([N_HIDDEN_2, 1], -0.003, 0.003))
        B3_c = tf.Variable(tf.random_uniform([1], -0.003, 0.003))

        H1_c = tf.nn.softplus(tf.matmul(critic_state_in, W1_c) + B1_c)
        H2_c = tf.nn.tanh(tf.matmul(H1_c, W2_c) + tf.matmul(critic_action_in, W2_action_c) + B2_c)

        critic_q_model = tf.matmul(H2_c, W3_c) + B3_c

        return W1_c, B1_c, W2_c, W2_action_c, B2_c, W3_c, B3_c, critic_q_model, critic_state_in, critic_action_in

    def train_critic(self, state_t_batch, action_batch, y_i_batch):
        self.sess.run(self.optimizer,
                      feed_dict={self.critic_state_in: state_t_batch, self.critic_action_in: action_batch,
                                 self.q_value_in: y_i_batch})

    def evaluate_target_critic(self, state_t_1, action_t_1):
        return self.sess.run(self.t_critic_q_model,
                             feed_dict={self.t_critic_state_in: state_t_1, self.t_critic_action_in: action_t_1})

    def compute_delQ_a(self, state_t, action_t):
        return self.sess.run(self.action_gradients,
                             feed_dict={self.critic_state_in: state_t, self.critic_action_in: action_t})

    def update_target_critic(self):
        self.sess.run([
            self.t_W1_c.assign(tf.mul(self.TAU, self.W1_c) + tf.mul(self.TAU_, self.t_W1_c)),
            self.t_B1_c.assign(tf.mul(self.TAU, self.B1_c) + tf.mul(self.TAU_, self.t_B1_c)),
            self.t_W2_c.assign(tf.mul(self.TAU, self.W2_c) + tf.mul(self.TAU_, self.t_W2_c)),
            self.t_W2_action_c.assign(tf.mul(self.TAU, self.W2_action_c) + tf.mul(self.TAU_, self.t_W2_action_c)),
            self.t_B2_c.assign(tf.mul(self.TAU, self.B2_c) + tf.mul(self.TAU_, self.t_B2_c)),
            self.t_W3_c.assign(tf.mul(self.TAU, self.W3_c) + tf.mul(self.TAU_, self.t_W3_c)),
            self.t_B3_c.assign(tf.mul(self.TAU, self.B3_c) + tf.mul(self.TAU_, self.t_B3_c))])
