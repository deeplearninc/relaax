import tensorflow as tf
import numpy as np
import math
import os

import game_ac_network
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork
from a3c_training_thread import A3CTrainingThread
from rmsprop_applier import RMSPropApplier

import base64
import json
import io


class Trainer:
    def __init__(self, params, target='', global_device='', local_device=''):
        self.params = params

        self._target = target

        kernel = "/cpu:0"
        if params.use_GPU:
            kernel = "/gpu:0"

        self._global_device = global_device + kernel
        self._local_device = local_device + kernel

        self._initial_learning_rate = None  # assign by static method log_uniform in initialize method
        self.global_t = 0                   # initial global steps count --> can be init via checkpoint
        self.training_threads = []          # Agent's Threads --> it's defined and assigned in initialize

        self.CHECKPOINT_DIR = None
        self.sess = self._initialize()

        self.frameDisplayQueue = None  # frame accumulator for state, cuz state = 4 consecutive frames
        self.display = False           # Becomes True when the Client initiates display session
        # Thread index for display == -1, but in initialize == -2 (for LSTM only)

    @staticmethod
    def _log_uniform(lo, hi, rate):
        log_lo = math.log(lo)
        log_hi = math.log(hi)
        v = log_lo * (1 - rate) + log_hi * rate
        return math.exp(v)

    def _initialize(self):
        self._initial_learning_rate = self._log_uniform(
            self.params.INITIAL_ALPHA_LOW,
            self.params.INITIAL_ALPHA_HIGH,
            self.params.INITIAL_ALPHA_LOG_RATE
        )
        with tf.device(self._global_device):
            self.global_network = game_ac_network.make_shared_network(self.params, -1)
            self.display_network = game_ac_network.make_full_network(self.params, -2)

        self._sync_display_network = game_ac_network.assign_vars(self.display_network, self.global_network)

        learning_rate_input = tf.placeholder("float")

        grad_applier = RMSPropApplier(learning_rate=learning_rate_input,
                                      decay=self.params.RMSP_ALPHA,
                                      momentum=0.0,
                                      epsilon=self.params.RMSP_EPSILON,
                                      clip_norm=self.params.GRAD_NORM_CLIP,
                                      device=self._local_device)

        for i in range(self.params.threads_cnt):
            training_thread = A3CTrainingThread(self.params, i, self.global_network,
                                                self._initial_learning_rate,
                                                learning_rate_input,
                                                grad_applier, self.params.max_global_step,
                                                self._local_device)
            self.training_threads.append(training_thread)

        variables = [(tf.is_variable_initialized(v), tf.initialize_variables([v])) for v in tf.all_variables()]

        # prepare session
        sess = tf.Session(
            target=self._target,
            config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        )

        for initialized, initialize in variables:
            if not sess.run(initialized):
                sess.run(initialize)

        return sess

    def getAction(self, message):
        thread_index = int(message['thread_index'])
        state = json.loads(message['state'], object_hook=ndarray_decoder)
        # state = state.astype(np.float32)
        # state *= (1.0 / 255.0)

        if thread_index == -1:
            return self.playing(state), -1

        training_thread = self.training_threads[thread_index]

        training_thread.update_state(state)
        state = training_thread.frameQueue

        if training_thread.episode_t == self.params.episode_len:
            self._update_global(training_thread.terminal_end, thread_index, state)

            if training_thread.terminal_end:
                training_thread.terminal_end = False

            training_thread.episode_t = 0

        if training_thread.episode_t == 0:
            # reset accumulated gradients
            self.sess.run(training_thread.reset_gradients)
            # copy weights from shared to local
            self.sess.run(training_thread.sync)

            training_thread.states = []
            training_thread.actions = []
            training_thread.rewards = []
            training_thread.values = []

            if self.params.use_LSTM:
                training_thread.start_lstm_state = training_thread.local_network.lstm_state_out

        pi_, value_ = training_thread.local_network.run_policy_and_value(self.sess, state)
        action = training_thread.choose_action(pi_)

        training_thread.states.append(state)
        training_thread.actions.append(action)
        training_thread.values.append(value_)

        if (thread_index == 0) and (training_thread.local_t % 100) == 0:
            print("pi=", pi_)
            print(" V=", value_)

        return action, thread_index

    def _update_global(self, terminal, thread_index, state):
        training_thread = self.training_threads[thread_index]
        R = 0.0
        if not terminal:
            R = training_thread.local_network.run_value(self.sess, state)

        training_thread.actions.reverse()
        training_thread.states.reverse()
        training_thread.rewards.reverse()
        training_thread.values.reverse()

        batch_si = []
        batch_a = []
        batch_td = []
        batch_R = []

        # compute and accumulate gradients
        for (ai, ri, si, Vi) in zip(training_thread.actions,
                                    training_thread.rewards,
                                    training_thread.states,
                                    training_thread.values):
            R = ri + self.params.GAMMA * R
            td = R - Vi
            a = np.zeros([self.params.action_size])
            a[ai] = 1

            batch_si.append(si)
            batch_a.append(a)
            batch_td.append(td)
            batch_R.append(R)

        if self.params.use_LSTM:
            batch_si.reverse()
            batch_a.reverse()
            batch_td.reverse()
            batch_R.reverse()

            self.sess.run(training_thread.accum_gradients,
                          feed_dict={
                              training_thread.local_network.s: batch_si,
                              training_thread.local_network.a: batch_a,
                              training_thread.local_network.td: batch_td,
                              training_thread.local_network.r: batch_R,
                              training_thread.local_network.initial_lstm_state:
                                  training_thread.start_lstm_state,
                              training_thread.local_network.step_size: [len(batch_a)]})
        else:
            self.sess.run(training_thread.accum_gradients,
                          feed_dict={
                              training_thread.local_network.s: batch_si,
                              training_thread.local_network.a: batch_a,
                              training_thread.local_network.td: batch_td,
                              training_thread.local_network.r: batch_R})

        cur_learning_rate = training_thread.anneal_learning_rate(self.global_t)

        self.sess.run(training_thread.apply_gradients,
                      feed_dict={training_thread.learning_rate_input: cur_learning_rate})

        if (thread_index == 0) and (training_thread.local_t % 100) == 0:
            print("TIMESTEP", training_thread.local_t)

    def addEpisode(self, message):
        thread_index = int(message['thread_index'])
        reward = message['reward']
        terminal = bool(message['terminal'])

        return_json = {'thread_index': thread_index,
                       'terminal': terminal,
                       'score': 0,
                       'stop_training': False}

        if thread_index == -1:
            if terminal:
                self.display = False
            return return_json

        training_thread = self.training_threads[thread_index]
        training_thread.episode_reward += reward

        # clip reward
        training_thread.rewards.append(np.clip(reward, -1, 1))

        training_thread.local_t += 1
        training_thread.episode_t += 1
        self.global_t += 1

        if self.global_t > self.params.max_global_step:
            # self.sio.emit('stop training', {}, room=self.session, namespace=self.namespace)
            return_json['stop_training'] = True
            return return_json

        if terminal:
            training_thread.terminal_end = True
            print("score=", training_thread.episode_reward)

            return_json['score'] = training_thread.episode_reward
            training_thread.episode_reward = 0

            if self.params.use_LSTM:
                training_thread.local_network.reset_state()

            training_thread.episode_t = self.params.episode_len

        return return_json

    def playing(self, frame):
        if not self.display:
            self.sess.run(self._sync_display_network)

        self.update_play_state(frame)
        state = self.frameDisplayQueue

        pi_values = self.display_network.run_policy(self.sess, state)
        action = self.training_threads[0].choose_action(pi_values)
        return action

    def update_play_state(self, frame):
        if self.display:
            self.frameDisplayQueue = np.append(self.frameDisplayQueue[:, :, 1:], frame, axis=2)
        else:
            self.frameDisplayQueue = np.stack((frame, frame, frame, frame), axis=2)
            self.display = True


def ndarray_decoder(dct):
    """Decoder from base64 to np.ndarray for big arrays(states)"""
    if isinstance(dct, dict) and 'b64npz' in dct:
        output = io.BytesIO(base64.b64decode(dct['b64npz']))
        output.seek(0)
        return np.load(output)['obj']
    return dct
