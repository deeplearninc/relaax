import sys
sys.path.append('../../server')

from uuid import uuid4
from flask import Flask, render_template, session  # request
from flask_socketio import SocketIO, emit, join_room, leave_room, close_room  # rooms, disconnect

import shared
import models.ale_model


app = Flask(__name__)
app.config['SECRET_KEY'] = 'TotalRecall'
socketio = SocketIO(app)
n_worker = None
server = None


def main(n_worker_):
    global n_worker
    global server
    n_worker = n_worker_
    server = shared.worker(n_worker_)
    return app


class ModelRunner(object):

    def __init__(self):
        self.models = {}

    def startModel(self, name, sio, session, namespace='/rlmodels'):
        self.stopModel(session)  # remove old instance of this model
        print("Loading model: " + name)
        self.models[session] = models.ale_model.AleModel(sio, session, namespace)
        self.models[session].start()

    def stopModel(self, session):
        if session in self.models:
            print("Removing model for: " + session)
            model = self.models.pop(session)
            print("Stopping model for: " + session)
            model.stop()

    def getModel(self, session):
        return self.models[session]


modelRunner = ModelRunner()


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect', namespace='/rlmodels')
def on_connect():
    session['id'] = str(uuid4())
    print('Client connected: ' + session['id'])
    emit('session id', {'session_id': session['id']})


@socketio.on('join', namespace='/rlmodels')
def join(message):
    join_room(message['room'])
    print('Client joined room: ' + message['room'])
    emit('join ack', {'room': message['room']}, room=session['id'])


@socketio.on('create model', namespace='/rlmodels')
def on_create_model(message):
    print("Creating model for: " + session['id'])
    modelRunner.startModel(message['model_name'], socketio, session['id'])


@socketio.on('get params', namespace='/rlmodels')
def on_get_params(message):
    print("Retrieve parameters for algorithm: " + message['algo_name'])
    model = modelRunner.getModel(session['id'])
    if model is not None:
        model.init_params(message['algo_name'])
    else:
        print('Error in on_get_params')
        emit('stop training ack', {}, room=session['id'])


@socketio.on('init model', namespace='/rlmodels')
def on_init_model(message):
    print("Initialize model algorithm with received parameters")
    model = modelRunner.getModel(session['id'])
    if model is not None:
        model.init_model(
            message,
            target=server.target,
            global_device=shared.ps_device(),
            local_device=shared.worker_device(n_worker)
        )
    else:
        print('Error in on_get_params')
        emit('stop training ack', {}, room=session['id'])


@socketio.on('get action', namespace='/rlmodels')
def on_get_action(message):
    # print("Get action for: " + session['id'])
    model = modelRunner.getModel(session['id'])
    if model is not None:
        emit('get action ack', {'action': model.getAction(message)}, room=session['id'])
    else:
        emit('get action error', {'data': 'no model found'}, room=session['id'])


@socketio.on('episode', namespace='/rlmodels')
def on_episode(message):
    # print("Episode for: " + session['id'])
    model = modelRunner.getModel(session['id'])
    if model is not None:
        model.addEpisode(message)
        # emit('episode ack', {}, room=session['id'])
    else:
        emit('get action error', {'data': 'no model found'}, room=session['id'])


@socketio.on('stop training', namespace='/rlmodels')
def on_stop_training():
    print("Stop training for: " + session['id'])
    model = modelRunner.getModel(session['id'])
    if model is not None:
        model.stop()
        emit('stop training ack', {}, room=session['id'])
    else:
        emit('get action error', {'data': 'no model found'}, room=session['id'])


@socketio.on('disconnect', namespace='/rlmodels')
def on_disconnect():
    if 'id' in session:
        model = modelRunner.getModel(session['id'])
        if hasattr(model.__class__, 'saveModel') and callable(getattr(model.__class__, 'saveModel')):
            print('Saving model for the room: ' + session['id'])
            model.saveModel(disconnect=True)
        print('Removing client from the room: ' + session['id'])
        leave_room(session['id'])
        close_room(session['id'])
        modelRunner.stopModel(session['id'])
    print('Client disconnected: ' + session['id'])


if __name__ == '__main__':
    socketio.run(app)
