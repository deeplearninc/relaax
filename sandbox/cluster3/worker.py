from __future__ import print_function

import sys
sys.path.append('../../server')

import base64
import flask
import flask_socketio
import grpc
import io
import json
import logging
import numpy
import os
import random

import algorithms.a3c.params
import algorithms.a3c.master
import algorithms.a3c.trainer

logging.basicConfig(
    format='%(asctime)s:%(levelname)s: %(message)s',
    level=logging.WARNING
)

app = flask.Flask(__name__)
app.config['SECRET_KEY'] = 'TotalRecall'
socketio = flask_socketio.SocketIO(app)
log_dir = None
master = algorithms.a3c.master.Stub(grpc.insecure_channel('localhost:50051'))


def main(n_worker_):
    global log_dir

    params = algorithms.a3c.params.Params()

    lstm_str = ''
    if params.use_LSTM:
        lstm_str = 'lstm_'
    log_dir = 'logs/boxing_a3c_%s%dthreads/worker_%d' % (lstm_str, 1, n_worker_)

    return app


class _Trainers(object):
    def __init__(self):
        self._params = algorithms.a3c.params.Params()
        self._trainers = {}

    def create_current(self):
        self._trainers[flask.request.sid] = algorithms.a3c.trainer.Trainer(
            params=self._params,
            master=master,
            log_dir=log_dir
        )

    def remove_current(self):
        if flask.request.sid in self._trainers:
            del self._trainers[flask.request.sid]

    def get_current(self):
        if flask.request.sid in self._trainers:
            return self._trainers[flask.request.sid]
        return None


_trainers = _Trainers()


@app.route('/')
def index():
    return flask.render_template('index.html')


@socketio.on('connect', namespace='/rlmodels')
def on_connect():
    logging.info('%d %s on_connect', os.getpid(), flask.request.sid)
    _trainers.create_current()
    _emit('connected')


@socketio.on('act', namespace='/rlmodels')
def on_act(state_dump):
    logging.info('%d %s on_act', os.getpid(), flask.request.sid)
    trainer = _trainers.get_current()
    if trainer is None:
        _emit('error', 'no trainer found')
        return
    state = json.loads(state_dump, object_hook=_ndarray_decoder)
    _emit('act', trainer.act(state))


@socketio.on('reward_and_reset', namespace='/rlmodels')
def on_reward_and_reset(reward):
    logging.info('%d %s on_reward_and_reset', os.getpid(), flask.request.sid)
    trainer = _trainers.get_current()
    if trainer is None:
        _emit('error', 'no trainer found')
        return
    score = trainer.reward_and_reset(reward)
    if score is None:
        flask_socketio.disconnect()
        return
    _emit('reset', score)


@socketio.on('reward_and_act', namespace='/rlmodels')
def on_reward_and_act(reward, state_dump):
    logging.info('%d %s on_reward_and_act', os.getpid(), flask.request.sid)
    trainer = _trainers.get_current()
    if trainer is None:
        _emit('error', 'no trainer found')
        return
    state = json.loads(state_dump, object_hook=_ndarray_decoder)
    action = trainer.reward_and_act(reward, state)
    if action is None:
        flask_socketio.disconnect()
        return
    _emit('act', action)


@socketio.on('disconnect', namespace='/rlmodels')
def on_disconnect():
    logging.info('%d %s on_disconnect', os.getpid(), flask.request.sid)
    _trainers.remove_current()


def _emit(verb, *args):
    flask_socketio.emit(verb, *args, room=flask.request.sid)


def _ndarray_decoder(dct):
    """Decoder from base64 to numpy.ndarray for big arrays(states)"""
    if isinstance(dct, dict) and 'b64npz' in dct:
        output = io.BytesIO(base64.b64decode(dct['b64npz']))
        output.seek(0)
        return numpy.load(output)['obj']
    return dct


if __name__ == '__main__':
    socketio.run(app)
