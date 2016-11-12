from __future__ import print_function

import tensorflow as tf
import numpy as np

import game_ac_network
import worker

import base64
import json
import io


class Trainer:
    def __init__(self, params, master, log_dir='.'):
        kernel = "/cpu:0"
        if params.use_GPU:
            kernel = "/gpu:0"

        with tf.device(kernel):
            self.display_network = game_ac_network.make_full_network(params, -2)

        self._worker = worker.Factory(
            params=params,
            master=master,
            local_device=kernel,
            get_session=lambda: self.sess,
            add_summary=lambda summary, step:
                self._summary_writer.add_summary(summary, step)
        )()

        variables = [(tf.is_variable_initialized(v), tf.initialize_variables([v])) for v in tf.all_variables()]

        # prepare session
        self.sess = tf.Session()

        self._summary_writer = tf.train.SummaryWriter(log_dir, self.sess.graph)

        for initialized, initialize in variables:
            if not self.sess.run(initialized):
                self.sess.run(initialize)

    def getAction(self, message):
        state = json.loads(message['state'], object_hook=_ndarray_decoder)
        return self._worker.act(state), 0

    def addEpisode(self, message):
        reward = message['reward']
        terminal = bool(message['terminal'])

        score, stop_training = self._worker.on_episode(reward, terminal)

        return {
            'terminal': terminal,
            'score': score,
            'stop_training': stop_training
        }


def _ndarray_decoder(dct):
    """Decoder from base64 to np.ndarray for big arrays(states)"""
    if isinstance(dct, dict) and 'b64npz' in dct:
        output = io.BytesIO(base64.b64decode(dct['b64npz']))
        output.seek(0)
        return np.load(output)['obj']
    return dct
