from __future__ import print_function

import sys
sys.path.append('../../server')

import flask
import flask_socketio
import grpc
import json
import logging
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
        return self._trainers[flask.request.sid]

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
    trainer = _trainers.create_current()
    _emit('connected', {'server_pid': os.getpid()})

@socketio.on('get action', namespace='/rlmodels')
def on_get_action(message):
    logging.info('%d %s on_get_action', os.getpid(), flask.request.sid)
    trainer = _trainers.get_current()
    if trainer is not None:
        _emit('get action ack', {'action': trainer.getAction(message)})
    else:
        _emit('get action error', {'data': 'no trainer found'})


@socketio.on('episode', namespace='/rlmodels')
def on_episode(message):
    logging.info('%d %s on_episode', os.getpid(), flask.request.sid)
    trainer = _trainers.get_current()
    if trainer is not None:
        data = trainer.addEpisode(message)
        _emit('episode ack', json.dumps(data))
    else:
        _emit('get action error', {'data': 'no trainer found'})


@socketio.on('stop training', namespace='/rlmodels')
def on_stop_training():
    logging.info('%d %s on_stop_training', os.getpid(), flask.request.sid)
    trainer = _trainers.get_current()
    if trainer is not None:
        _trainers.remove_current()
        _emit('stop training ack', {})
    else:
        _emit('get action error', {'data': 'no trainer found'})


@socketio.on('disconnect', namespace='/rlmodels')
def on_disconnect():
    logging.info('%d %s on_disconnect', os.getpid(), flask.request.sid)
    _trainers.remove_current()


def _emit(verb, json):
    flask_socketio.emit(verb, json, room=flask.request.sid)


if __name__ == '__main__':
    socketio.run(app)
