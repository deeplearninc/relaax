from __future__ import print_function

import tensorflow as tf
import numpy as np

import game_ac_network
import worker

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

        with tf.device(global_device + kernel):
            self.global_network = game_ac_network.make_shared_network(params, -1)
            self.display_network = game_ac_network.make_full_network(params, -2)

        new_worker = worker.Factory(
            params=params,
            global_network=self.global_network,
            local_device=local_device + kernel,
            get_session=lambda: self.sess,
            add_summary=lambda summary, step:
                self._summary_writer.add_summary(summary, step)
        )

        self._sync_display_network = game_ac_network.assign_vars(self.display_network, self.global_network)

        self._workers = [new_worker() for _ in xrange(params.threads_cnt)]

        variables = [(tf.is_variable_initialized(v), tf.initialize_variables([v])) for v in tf.all_variables()]

        # prepare session
        self.sess = tf.Session(
            target=self._target,
            config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        )

        self._summary_writer = tf.train.SummaryWriter(log_dir, self.sess.graph)

        for initialized, initialize in variables:
            if not self.sess.run(initialized):
                self.sess.run(initialize)

        self.frameDisplayQueue = None  # frame accumulator for state, cuz state = 4 consecutive frames
        self.display = False           # Becomes True when the Client initiates display session

    def getAction(self, message):
        thread_index = int(message['thread_index'])
        state = json.loads(message['state'], object_hook=_ndarray_decoder)
        # print('-' * 20, repr(state))
        # state = state.astype(np.float32)
        # state *= (1.0 / 255.0)

        if thread_index == -1:
            return self.playing(state), -1

        return self._workers[thread_index].act(state), thread_index

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

        score, stop_training = self._workers[thread_index].on_episode(reward, terminal)

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
