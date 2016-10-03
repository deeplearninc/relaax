import tensorflow as tf
from dqn import DQN

import numpy as np

import random
import base64
import json
import io


class Trainer:
    def __init__(self, params):
        self.params = params

        self.device = "/cpu:0"
        if params.use_GPU:
            self.device = "/gpu:0"

        self.agent = DQN(self.params)
        self.saver = tf.train.Saver()

        self.sess = self.initialize()

        self.randomRestart = 0          # 1st part of training process
        self.replayStart = 0            # 2nd part of training process
        self.trainSteps = -1            # 3rd part of training process

        self.reward = None      # auxiliary param to store last reward
        self.terminal = None    # auxiliary param to store last terminal

        self.sample_success = 0
        self.sample_failure = 0
        self.total_loss = 0
        self.gameReward = 0

    def initialize(self):
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        self.params.start_from = self.agent.restore(self.saver, sess)
        print "Start training from step: ", self.params.start_from

        return sess

    def getAction(self, message):
        state = json.loads(message['state'], object_hook=ndarray_decoder)
        state = state.astype(np.float32)
        state *= (1.0 / 255.0)

        if 'thread_index' in message.keys() and message['thread_index'] == -1:
            self.agent.game_buffer.add(state)
            return self.agent.observe(self.agent.eps, self.sess, True), -1

        if self.trainSteps != -1:
            # annealing learning rate
            lr = self.agent.trainEps(self.trainSteps)
            a = self.agent.observe(lr, self.sess)
            action = np.zeros(self.agent.num_actions)
            action[a] = 1.0
            self.trainSteps += 1
            old_state = self.agent.buffer.getState()
            self.agent.buffer.add(state)
            self.agent.memory.add(old_state, action, self.reward, self.agent.buffer.getState(), self.terminal)
            return a

        if self.replayStart != 0:
            if self.replayStart == self.agent.replay_start_size:
                self.trainSteps += self.params.start_from + 1
                print "\nstart training..."
            a = random.randrange(self.agent.num_actions)
            action = np.zeros(self.agent.num_actions)
            action[a] = 1.0
            self.replayStart += 1
            if self.reward is not None:
                old_state = self.agent.buffer.getState()
                self.agent.buffer.add(state)
                self.agent.memory.add(old_state, action, self.reward, self.agent.buffer.getState(), self.terminal)
            return a

        self.agent.buffer.add(state)
        self.randomRestart += 1
        if self.randomRestart == self.agent.random_starts:
            self.replayStart += 1
            print "starting %d random plays to populate replay memory" % self.agent.replay_start_size
        return random.randrange(self.agent.num_actions)

    def addEpisode(self, message):
        reward = message['reward']
        self.terminal = bool(message['terminal'])
        self.gameReward += reward

        if 'thread_index' in message.keys() and message['thread_index'] == -1:
            return_json = {'terminal': self.terminal,
                           'score': self.gameReward,
                           'stop_training': False,
                           'thread_index': -1}
            return return_json

        return_json = {'terminal': self.terminal,
                       'score': self.gameReward,
                       'stop_training': False}

        self.reward = np.clip(reward, -1.0, 1.0)
        if self.terminal:
            print("score=", self.gameReward)
            self.gameReward = 0

        if self.trainSteps != -1:
            if len(self.agent.memory) > self.agent.batch_size and self.trainSteps % self.agent.update_freq == 0:
                self.sample_success, self.sample_failure, loss = self.agent.doMinibatch(self.sess,
                                                                                        self.sample_success,
                                                                                        self.sample_failure)
                self.total_loss += loss

            if self.trainSteps % self.agent.steps == 0:
                self.agent.copy_weights(self.sess)

            if self.trainSteps % self.agent.save_weights == 0:
                self.agent.save(self.saver, self.sess, self.trainSteps)

            if self.trainSteps % self.agent.batch_size == 0:
                avg_loss = self.total_loss / self.agent.batch_size
                print "\nTraining step: ", self.trainSteps, \
                      "\nmemory size: ", len(self.agent.memory), \
                      "\nSample successes: ", self.sample_success, \
                      "\nSample failures: ", self.sample_failure, \
                      "\nAverage batch loss: ", avg_loss
                self.total_loss = 0

        return return_json

    def saveModel(self, disconnect=False):
        if self.trainSteps != -1:
            self.agent.save(self.saver, self.sess, self.trainSteps, disconnect)


def ndarray_decoder(dct):
    """Decoder from base64 to np.ndarray for big arrays(states)"""
    if isinstance(dct, dict) and 'b64npz' in dct:
        output = io.BytesIO(base64.b64decode(dct['b64npz']))
        output.seek(0)
        return np.load(output)['obj']
    return dct
