import sys
sys.path.append('../../server')

import json
import flask
import flask_socketio
from flask_socketio import emit # rooms, disconnect

import shared
import models.ale_model

import algorithms.a3c.params
import algorithms.a3c.trainer


app = flask.Flask(__name__)
app.config['SECRET_KEY'] = 'TotalRecall'
socketio = flask_socketio.SocketIO(app)
n_worker = None
server = None


def main(n_worker_):
    global n_worker
    global server
    n_worker = n_worker_
    server = shared.worker(n_worker_)
    return app


class _ModelSet(object):
    def __init__(self):
        self._params = algorithms.a3c.params.Params()
        self._models = {}

    def createCurrent(self):
        self._models[flask.request.sid] = models.ale_model.AleModel(
            self._params,
            algorithms.a3c.trainer.Trainer
        )
        return self._models[flask.request.sid]

    def removeCurrent(self):
        if flask.request.sid in self._models:
            del self._models[flask.request.sid]

    def getCurrent(self):
        if flask.request.sid in self._models:
            return self._models[flask.request.sid]
        return None


modelSet = _ModelSet()


@app.route('/')
def index():
    return flask.render_template('index.html')


@socketio.on('connect', namespace='/rlmodels')
def on_connect():
    print('on connect: ' + flask.request.sid)
    flask.session['id'] = flask.request.sid
    model = modelSet.createCurrent()
    model.init_model(
        target=server.target,
        global_device=shared.ps_device(),
        local_device=shared.worker_device(n_worker)
    )
    emit('model is ready', {'threads_cnt': model.threads_cnt()}, room=flask.session['id'])


@socketio.on('get action', namespace='/rlmodels')
def on_get_action(message):
    # print("Get action for: " + flask.session['id'])
    model = modelSet.getCurrent()
    if model is not None:
        emit('get action ack', {'action': model.getAction(message)}, room=flask.session['id'])
    else:
        emit('get action error', {'data': 'no model found'}, room=flask.session['id'])


@socketio.on('episode', namespace='/rlmodels')
def on_episode(message):
    # print("Episode for: " + flask.session['id'])
    model = modelSet.getCurrent()
    if model is not None:
        data = model.addEpisode(message)
        emit('episode ack', json.dumps(data), room=flask.session['id'])
    else:
        emit('get action error', {'data': 'no model found'}, room=flask.session['id'])


@socketio.on('stop training', namespace='/rlmodels')
def on_stop_training():
    print("Stop training for: " + flask.session['id'])
    model = modelSet.getCurrent()
    if model is not None:
        modelSet.removeCurrent()
        emit('stop training ack', {}, room=flask.session['id'])
    else:
        emit('get action error', {'data': 'no model found'}, room=flask.session['id'])


@socketio.on('disconnect', namespace='/rlmodels')
def on_disconnect():
    if 'id' in flask.session:
        print('Removing client from the room: ' + flask.session['id'])
        modelSet.removeCurrent()
    print('Client disconnected: ' + flask.session['id'])


if __name__ == '__main__':
    socketio.run(app)
