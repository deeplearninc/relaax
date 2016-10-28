from __future__ import print_function

import base64
import io
import json
import nonblock
import numpy
import sys
import threading
import socketIO_client


class _NDArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            output = io.BytesIO()
            numpy.savez_compressed(output, obj=obj)
            return {'b64npz': base64.b64encode(output.getvalue())}
        return json.JSONEncoder.default(self, obj)


class ServerAPI(socketIO_client.LoggingNamespace):
    def __init__(self, cfg, factory, *args, **kwargs):
        socketIO_client.LoggingNamespace.__init__(self, *args, **kwargs)

        self.inputKey = nonblock.bgread(sys.stdin)   # for Display --> standard console input interception
        self.play_thread = None     # Display thread --> need for further deleting
        self.gamePlayedList = None  # Games played accumulator for current session (can holds all threads)
        self.cfg = cfg              # Parameters to setup the environment and training process
        self.factory = factory
        self.gameList = []          # List of running games, if no parallel agents -> list holds one element

    def on_model_is_ready(self, *args):
        print('on_model_is_ready', args)

        for i in xrange(self.cfg.threads_cnt):
            self.gameList.append(self.factory.new_env(113 * i))

        self.gamePlayedList = numpy.zeros(self.cfg.threads_cnt, dtype=int)

        for i in xrange(self.cfg.threads_cnt):
            print('Agent\'s game is created with index:', i)
            self.emit('get action', {'thread_index': i, 'state': self.dump_state(i)})

    def on_get_action_ack(self, *args):
        # if more than one agent trains -> server returns tuple(list) with [action_num, thread_num]
        if hasattr(args[0]['action'], '__getitem__'):
            action = args[0]['action'][0]
            index = args[0]['action'][1]
        else:
            action = args[0]['action']
            index = 0

        # receive game result
        reward = self.gameList[index].act(action)
        terminal = self.gameList[index].terminal

        if terminal and index != -1:
            self.gameList[index].reset()

        self.emit('episode', {
            'thread_index': index,
            'reward': reward,
            'terminal': terminal
        })

    def on_episode_ack(self, *args):
        episode_params = json.loads(args[0])
        if episode_params['stop_training']:
            self.emit('stop training')
            return

        if episode_params.__contains__('thread_index'):
            index = episode_params['thread_index']

            if episode_params['terminal']:
                if index == -1:
                    self.stop_play_thread()
                    return None
                self.gamePlayedList[index] += 1
                print("Score for thread", index, "at game", self.gamePlayedList[index],
                      "=", int(episode_params['score']))

            self.emit('get action', {
                'thread_index': index,
                'state': self.dump_state(index)
            })
        else:
            if episode_params['terminal']:
                self.gamePlayedList += 1
                print("Score for agent at game", self.gamePlayedList, "=", episode_params['score'])

            self.emit('get action', {
                'state': self.dump_state(0)
            })

        key = self.inputKey.data
        if key != '':
            print(key)
            if key[-2] == "d":
                print('Playing game for training algorithm at current step...')
                self.gameList.append(self.factory.new_display_env(0))
                self.play_thread = threading.Thread(target=self.emit('get action', {
                    'thread_index': -1,
                    'state': self.dump_state(-1)
                }))
                self.play_thread.start()
            self.inputKey.isFinished = True
            self.inputKey = nonblock.bgread(sys.stdin)

    def on_stop_training_ack(self, *args):
        print('on_stop_training_ack', args)
        self.emit('disconnect', {})

    def stop_play_thread(self):
        raise NotImplementedError()


    def dump_state(self, i):
        state = self.gameList[i].state()
        # print('-' * 20, repr(state))
        return json.dumps(state, cls=_NDArrayEncoder)
