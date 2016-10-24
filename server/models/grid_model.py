from base_model import BaseModel
import importlib

import numpy as np
import random
import time
import json

from collections import deque
BUFFER_MAXLEN = 80


class GridModel(BaseModel):
    def __init__(self, *args, **kwargs):
        super(GridModel, self).__init__(*args, **kwargs)
        self.episodeQueue = deque(maxlen=BUFFER_MAXLEN)

        self.algo_name = None   # assign via get_params event in init_params method
        self.params = None      # assign via get_params event in init_params method

        self.qvalue = None      # QLearn(64, 4)
        self.batchSize = None   # 40
        self.gamma = None       # 0.975

    def init_params(self, algo_name):
        module = importlib.import_module("algorithms." + algo_name + ".params")
        clazz = getattr(module, 'Params')
        self.params = clazz()  # get the instance of Params Class to perform
        self.algo_name = algo_name
        self.sio.emit('init params', json.dumps(self.params.default_params),
                      room=self.session, namespace=self.namespace)

    def init_model(self, message):
        print(message)
        params = json.loads(message)

        for param_name in params:
            if hasattr(self.params, param_name):
                setattr(self.params, param_name, params[param_name])
        # for Simplicity
        self.batchSize = self.params.batch_size
        self.gamma = self.params.gamma

        module = importlib.import_module("algorithms." + self.algo_name + "." + self.algo_name)
        clazz = getattr(module, super(GridModel, self).to_camelcase(self.algo_name))
        self.qvalue = clazz(self.params.grid_size, self.params.action_size)  # (64, 4)
        self.sio.emit('model is ready', {}, room=self.session, namespace=self.namespace)

    def getAction(self, message):
        state = json.loads(message['state'], object_hook=super(GridModel, self).ndarray_decoder)
        epsilon = float(message['epsilon'])

        print(epsilon)
        qval = self.qvalue.predict(state.reshape(1, 64), batch_size=1)
        print(qval)

        if random.random() < epsilon:  # choose random action
            action = np.random.randint(0, 4)
        else:  # choose best action from Q(s,a) values
            action = (np.argmax(qval))

        return action, 0

    def _trainModel(self):
        if len(self.episodeQueue) < self.episodeQueue.maxlen:
            return None

        # print("Train")
        minibatch = random.sample(self.episodeQueue, self.batchSize)
        # print(minibatch)
        X_train = []
        y_train = []
        for memory in minibatch:
            # Get max_Q(S',a)
            old_state, action, reward, new_state = memory
            # print(old_state)
            old_qval = self.qvalue.predict(old_state.reshape(1, 64), batch_size=1)
            # print(old_qval)
            newQ = self.qvalue.predict(new_state.reshape(1, 64), batch_size=1)
            maxQ = np.max(newQ)
            y = np.zeros((1, 4))
            y[:] = old_qval[:]
            if reward == -1:  # non-terminal state
                update = (reward + (self.gamma * maxQ))
            else:  # terminal state
                update = reward
            y[0][action] = update
            X_train.append(old_state.reshape(64,))
            y_train.append(y.reshape(4,))

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        # print("Game #: %s" % (i,))
        self.qvalue.fit(X_train, y_train, batch_size=self.batchSize, nb_epoch=1, verbose=1)

    def addEpisode(self, message):
        state = json.loads(message['state'], object_hook=super(GridModel, self).ndarray_decoder)
        action = int(message['action'])
        reward = int(message['reward'])
        new_state = json.loads(message['new_state'], object_hook=super(GridModel, self).ndarray_decoder)

        self.episodeQueue.append((state, action, reward, new_state))
        if len(self.episodeQueue) >= self.episodeQueue.maxlen:
            self._trainModel()

        self.sio.emit('episode ack', {}, room=self.session, namespace=self.namespace)

    def train(self):
        while not self.isStopped():
            time.sleep(10)
