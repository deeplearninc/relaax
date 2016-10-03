import tensorflow as tf
import random as rand
import numpy as np
from convnet import ConvNet
from buff import Buffer
from memory import Memory
import os


class DQN:
    def __init__(self, params):
        self.num_actions = params.action_size
        self.episodes = params.episodes
        self.steps = params.steps
        self.train_steps = params.max_global_step
        self.update_freq = params.episode_len
        self.save_weights = params.save_weights
        self.history_length = params.history_len
        self.discount = params.discount
        self.eps = params.init_eps
        self.eps_delta = (params.init_eps - params.final_eps) / params.final_eps_frame
        self.replay_start_size = params.replay_start_size
        self.eps_endt = params.final_eps_frame
        self.random_starts = params.random_starts
        self.batch_size = params.batch_size

        game_name = params.game_rom.split("-")
        self.ckpt_dir = 'checkpoints/' + game_name[0] + '_dqn'
        self.ckpt_file = self.ckpt_dir+'/'+game_name[0]

        self.global_step = tf.Variable(0, trainable=False)
        if params.lr_anneal:
            self.lr = tf.train.exponential_decay(params.lr, self.global_step, params.lr_anneal, 0.96, staircase=True)
        else:
            self.lr = params.lr

        self.buffer = Buffer(params)
        self.game_buffer = Buffer(params)
        self.memory = Memory(params.memory_size, self.batch_size)

        with tf.variable_scope("train") as self.train_scope:
            self.train_net = ConvNet(params, trainable=True)
        with tf.variable_scope("target") as self.target_scope:
            self.target_net = ConvNet(params, trainable=False)

        self.optimizer = tf.train.RMSPropOptimizer(self.lr, params.decay_rate, 0.0, self.eps)

        self.actions = tf.placeholder(tf.float32, [None, self.num_actions])
        self.q_target = tf.placeholder(tf.float32, [None])
        self.q_train = tf.reduce_max(tf.mul(self.train_net.y, self.actions), reduction_indices=1)
        self.diff = tf.sub(self.q_target, self.q_train)

        half = tf.constant(0.5)
        if params.clip_delta > 0:
            abs_diff = tf.abs(self.diff)
            clipped_diff = tf.clip_by_value(abs_diff, 0, 1)
            linear_part = abs_diff - clipped_diff
            quadratic_part = tf.square(clipped_diff)
            self.diff_square = tf.mul(half, tf.add(quadratic_part, linear_part))
        else:
            self.diff_square = tf.mul(half, tf.square(self.diff))

        if params.accumulator == 'sum':
            self.loss = tf.reduce_sum(self.diff_square)
        else:
            self.loss = tf.reduce_mean(self.diff_square)

        # back propagation with RMS loss
        self.task = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def trainEps(self, train_step):
        if train_step < self.eps_endt:
            return self.eps - train_step * self.eps_delta
        else:
            return self.eps_endt

    def observe(self, exploration_rate, sess, play=False):    # fixed --> return a
        if rand.random() < exploration_rate:
            a = rand.randrange(self.num_actions)
        else:
            x = self.buffer.getInput()
            if play:
                x = self.game_buffer.getInput()
            action_values = self.train_net.y.eval(feed_dict={self.train_net.x: x}, session=sess)
            a = np.argmax(action_values)
        return a

    def doMinibatch(self, sess, successes, failures):
        batch = self.memory.getSample()
        state = np.array([batch[i][0] for i in range(self.batch_size)]).astype(np.float32)
        actions = np.array([batch[i][1] for i in range(self.batch_size)]).astype(np.float32)
        rewards = np.array([batch[i][2] for i in range(self.batch_size)]).astype(np.float32)
        successes += np.sum(rewards == 1)
        next_state = np.array([batch[i][3] for i in range(self.batch_size)]).astype(np.float32)
        terminals = np.array([batch[i][4] for i in range(self.batch_size)]).astype(np.float32)

        failures += np.sum(terminals == 1)
        failures += np.sum(rewards == -1)   # my addition for specific games: boxing, pong, etc...
        q_target = self.target_net.y.eval(feed_dict={self.target_net.x: next_state}, session=sess)
        q_target_max = np.argmax(q_target, axis=1)
        q_target = rewards + ((1.0 - terminals) * (self.discount * q_target_max))

        (result, loss) = sess.run([self.task, self.loss],
                                  feed_dict={self.q_target: q_target,
                                             self.train_net.x: state,
                                             self.actions: actions})
        return successes, failures, loss

    def copy_weights(self, sess):
        for key in self.train_net.weights.keys():
            t_key = 'target/' + key.split('/', 1)[1]
            sess.run(self.target_net.weights[t_key].assign(self.train_net.weights[key]))

    def save(self, saver, sess, step, disconnect=False):
        if not os.path.exists(self.ckpt_dir):
            # os.mkdir(self.ckpt_dir)
            os.makedirs(self.ckpt_dir)

        saver.save(sess, self.ckpt_file, global_step=step)
        if disconnect:
            sess.close()
        
    def restore(self, saver, sess):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            tokens = ckpt.model_checkpoint_path.split("-")
            return int(tokens[1])
        return 0
