from __future__ import print_function

import tensorflow as tf
import numpy as np
import math

import game_ac_network
import worker
from rmsprop_applier import RMSPropApplier

import base64
import json
import io


class Trainer:
    def __init__(self, params, target='', global_device='', local_device='', log_dir='.'):
        self._target = target

        kernel = "/cpu:0"
        if params.use_GPU:
            kernel = "/gpu:0"

        self._workers = []          # Agent's Threads --> it's defined and assigned in initialize

        self.sess = self._initialize(params, global_device + kernel, local_device + kernel, log_dir)

        self.frameDisplayQueue = None  # frame accumulator for state, cuz state = 4 consecutive frames
        self.display = False           # Becomes True when the Client initiates display session

    def _initialize(self, params, global_device, local_device, log_dir):
        with tf.device(global_device):
            self.global_network = game_ac_network.make_shared_network(params, -1)
            self.display_network = game_ac_network.make_full_network(params, -2)

        self._sync_display_network = game_ac_network.assign_vars(self.display_network, self.global_network)

        learning_rate_input = tf.placeholder("float")

        grad_applier = RMSPropApplier(learning_rate=learning_rate_input,
                                      decay=params.RMSP_ALPHA,
                                      momentum=0.0,
                                      epsilon=params.RMSP_EPSILON,
                                      clip_norm=params.GRAD_NORM_CLIP,
                                      device=local_device)

        for i in range(params.threads_cnt):
            self._workers.append(worker.Worker(
                params,
                i,
                self.global_network,
                learning_rate_input,
                grad_applier,
                params.max_global_step,
                local_device
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

        self._summary_writer = tf.train.SummaryWriter(log_dir, sess.graph)

        for initialized, initialize in variables:
            if not sess.run(initialized):
                sess.run(initialize)

        return sess

    def getAction(self, message):
        thread_index = int(message['thread_index'])
        state = json.loads(message['state'], object_hook=_ndarray_decoder)
        # print('-' * 20, repr(state))
        # state = state.astype(np.float32)
        # state *= (1.0 / 255.0)

        if thread_index == -1:
            return self.playing(state), -1

        return self._workers[thread_index].act(self, state), thread_index

    def addEpisode(self, message):
        thread_index = int(message['thread_index'])
        reward = message['reward']
        terminal = bool(message['terminal'])

        if thread_index == -1:
            if terminal:
                self.display = False
            return {
                'thread_index': thread_index,
                'terminal': terminal,
                'score': 0,
                'stop_training': False
            }

        score, stop_training = self._workers[thread_index].on_episode(self, reward, terminal)

        return {
            'thread_index': thread_index,
            'terminal': terminal,
            'score': score,
            'stop_training': stop_training
        }

    def playing(self, frame):
        if not self.display:
            self.sess.run(self._sync_display_network)

        self.update_play_state(frame)

        return self._workers[0].choose_action(
            self.display_network.run_policy(self.sess, self.frameDisplayQueue)
        )

    def update_play_state(self, frame):
        if self.display:
            self.frameDisplayQueue = np.append(self.frameDisplayQueue[:, :, 1:], frame, axis=2)
        else:
            self.frameDisplayQueue = np.stack((frame, frame, frame, frame), axis=2)
            self.display = True


def _ndarray_decoder(dct):
    """Decoder from base64 to np.ndarray for big arrays(states)"""
    if isinstance(dct, dict) and 'b64npz' in dct:
        output = io.BytesIO(base64.b64decode(dct['b64npz']))
        output.seek(0)
        return np.load(output)['obj']
    return dct
