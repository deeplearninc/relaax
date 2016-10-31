from __future__ import print_function

import tensorflow as tf
import numpy as np
import math

import game_ac_network
from rmsprop_applier import RMSPropApplier
from tensorflow.python.training import slot_creator

import worker

import base64
import json
import io


class Trainer:
    def __init__(self, params, target='', global_device='', local_device='', log_dir='.'):
        self.params = params

        self._target = target

        kernel = "/cpu:0"
        if params.use_GPU:
            kernel = "/gpu:0"

        self._global_device = global_device + kernel
        self._local_device = local_device + kernel

        self._initial_learning_rate = None  # assign by static method log_uniform in initialize method
        self._workers = []          # Agent's Threads --> it's defined and assigned in initialize

        self._log_dir = log_dir

        self.sess = self._initialize()

        self.frameDisplayQueue = None  # frame accumulator for state, cuz state = 4 consecutive frames
        self.display = False           # Becomes True when the Client initiates display session

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

        self._sync_display_network = game_ac_network.assign_vars(self.display_network.get_vars(), self.global_network.get_vars())

        learning_rate_input = tf.placeholder("float")

        slots = _Slots("RMSPropApplier")

        with tf.device(self._local_device):
            for var in self.global_network.get_vars():
                slots.create_rms(var)
                slots.create_momentum(var)

        grad_applier = RMSPropApplier(learning_rate=learning_rate_input,
                                      decay=self.params.RMSP_ALPHA,
                                      momentum=0.0,
                                      epsilon=self.params.RMSP_EPSILON,
                                      clip_norm=self.params.GRAD_NORM_CLIP,
                                      device=self._local_device,
                                      name="RMSPropApplier",
                                      slots=slots,
                                      global_vars=self.global_network.get_vars())

        for i in range(self.params.threads_cnt):
            self._workers.append(worker.Worker(
                self.params,
                lambda: self.sess,
                i,
                self.global_network,
                self._initial_learning_rate,
                learning_rate_input,
                grad_applier,
                self.params.max_global_step,
                self._local_device
            ))

        self._episode_score = tf.placeholder(tf.int32)
        tf.scalar_summary('episode score', self._episode_score)

        self._summary = tf.merge_all_summaries()

        variables = [(tf.is_variable_initialized(v), tf.initialize_variables([v])) for v in tf.all_variables()]

        # prepare session
        sess = tf.Session(
            target=self._target,
            config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        )

        self._summary_writer = tf.train.SummaryWriter(self._log_dir, sess.graph)

        for initialized, initialize in variables:
            if not sess.run(initialized):
                sess.run(initialize)

        return sess

    def getAction(self, message):
        thread_index = int(message['thread_index'])
        state = json.loads(message['state'], object_hook=ndarray_decoder)

        if thread_index == -1:
            return self.playing(state), -1

        return self._workers[thread_index].act(state), thread_index

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

        worker = self._workers[thread_index]
        worker.episode_reward += reward

        # clip reward
        worker.rewards.append(np.clip(reward, -1, 1))

        worker.local_t += 1
        worker.episode_t += 1
        global_t = self.sess.run(self.global_network.increment_global_t)

        if global_t > self.params.max_global_step:
            # self.sio.emit('stop training', {}, room=self.session, namespace=self.namespace)
            return_json['stop_training'] = True
            return return_json

        if terminal:
            worker.terminal_end = True
            print("score=", worker.episode_reward)

            return_json['score'] = worker.episode_reward

            self._summary_writer.add_summary(
                self.sess.run(self._summary, feed_dict={
                    self._episode_score: worker.episode_reward
                }),
                global_t
            )

            worker.episode_reward = 0

            if self.params.use_LSTM:
                worker.local_network.reset_state()

            worker.episode_t = self.params.episode_len

        return return_json

    def playing(self, frame):
        if not self.display:
            self.sess.run(self._sync_display_network)

        self.update_play_state(frame)
        state = self.frameDisplayQueue

        pi_values = self.display_network.run_policy(self.sess, state)
        action = self._workers[0].choose_action(pi_values)
        return action

    def update_play_state(self, frame):
        if self.display:
            self.frameDisplayQueue = np.append(self.frameDisplayQueue[:, :, 1:], frame, axis=2)
        else:
            self.frameDisplayQueue = np.stack((frame, frame, frame, frame), axis=2)
            self.display = True


class _Slots(object):
    RMS = 'rms'
    MOMENTUM = 'momentum'

    def __init__(self, name):
        self._name = name
        self._slots = {}

    def create_rms(self, var):
        self._make_slot(
            var,
            self.RMS,
            lambda: slot_creator.create_slot(
                var,
                tf.constant(1.0, dtype=var.dtype, shape=var.get_shape()),
                self._name
            )
        )

    def create_momentum(self, var):
        self._make_slot(
            var,
            self.MOMENTUM,
            lambda: slot_creator.create_zeros_slot(var, self._name)
        )

    def get_rms(self, var):
        return self._get_slot(var, self.RMS)

    def get_momentum(self, var):
        return self._get_slot(var, self.MOMENTUM)

    def _make_slot(self, var, slot_name, factory):
        key = (var, slot_name)
        if key in self._slots:
            return
        self._slots[key] = factory()

    def _get_slot(self, var, slot_name):
        return self._slots.get((var, slot_name))


def ndarray_decoder(dct):
    """Decoder from base64 to np.ndarray for big arrays(states)"""
    if isinstance(dct, dict) and 'b64npz' in dct:
        output = io.BytesIO(base64.b64decode(dct['b64npz']))
        output.seek(0)
        return np.load(output)['obj']
    return dct
