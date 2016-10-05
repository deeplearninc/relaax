from __future__ import print_function
import numpy as np
import json
import io
import base64

from socketIO_client import SocketIO, LoggingNamespace
import logging
import threading
import sys

from time import sleep
from nonblock import bgread

from game_process import GameProcess as Game
from params import Params


logging.getLogger('requests').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)


class NDArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.savez_compressed(output, obj=obj)
            return {'b64npz': base64.b64encode(output.getvalue())}
        return json.JSONEncoder.default(self, obj)


class ServerAPI(LoggingNamespace):
    def __init__(self, *args, **kwargs):
        super(ServerAPI, self).__init__(*args, **kwargs)
        self.cfg = Params()         # instantiate the parameters to perform

        self.gameList = []          # List of running games, if no parallel agents -> list holds one element
        self.gamePlayedList = None  # Games played accumulator for current session (can holds all threads)

        self.inputKey = bgread(sys.stdin)   # for Display --> standard console input interception
        self.play_thread = None             # Display thread --> need for further deleting

    def on_session_id(self, *args):
        print('on_session_response', args)
        self.emit('join', {'room': args[0]['session_id']})

    def on_join_ack(self, *args):
        print('on_join_ack', args)
        self.emit('create model', {'model_name': self.cfg.args.scope})

    def on_model_is_allocated(self, *args):
        print('on_model_is_allocated', args)
        self.emit('get params', {'algo_name': self.cfg.args.algo})

    def on_init_params(self, *args):
        print('on_init_params', args)

        if args[0].__contains__('threads_cnt'):
            for i in range(self.cfg.threads_cnt):
                self.gameList.append(Game(113 * i, self.cfg.game_rom))
        else:
            self.gameList.append(Game(0, self.cfg.game_rom))
        self.cfg.action_size = self.gameList[0].real_action_size()  # Action size for the given game

        params = json.loads(args[0])
        for param_name in params:
            if hasattr(self.cfg, param_name):
                params[param_name] = getattr(self.cfg, param_name)

        print('Name of the target game:', self.cfg.game_rom)
        print('Action size for the target game:', self.cfg.action_size)
        self.emit('init model', json.dumps(params))

    def on_model_is_ready(self, *args):
        print('on_model_is_ready', args)

        if args[0].__contains__('threads_cnt'):
            self.gamePlayedList = np.zeros(args[0]['threads_cnt'], dtype=int)

            for i in range(args[0]['threads_cnt']):
                print('Game agent\'s thread is created with index:', i)
                threading.Thread(target=self.emit('get action',
                                                  {'thread_index': i,
                                                   'state': json.dumps(self.gameList[i].s_t, cls=NDArrayEncoder)}))
        else:
            self.gamePlayedList = 0
            print('Game is created for the training...')
            self.emit('get action', {'state': json.dumps(self.gameList[0].s_t, cls=NDArrayEncoder)})

    def on_get_action_ack(self, *args):
        # if more than one agent trains -> server returns tuple(list) with [action_num, thread_num]
        if hasattr(args[0]['action'], '__getitem__'):
            action = args[0]['action'][0]
            index = args[0]['action'][1]
        else:
            action = args[0]['action']
            index = 0

        # process game
        self.gameList[index].process(action)

        # receive game result
        reward = self.gameList[index].reward
        terminal = self.gameList[index].terminal

        if terminal and index != -1:
            self.gameList[index].reset()

        self.emit('episode', {'thread_index': index,
                              'reward': reward,
                              'terminal': terminal})

    def on_episode_ack(self, *args):
        episode_params = json.loads(args[0])
        if episode_params['stop_training']:
            self.emit('stop training')
            return

        if episode_params.__contains__('thread_index'):
            index = episode_params['thread_index']

            if episode_params['terminal']:
                if index == -1:
                    self.play_thread.join()
                    sleep(3)
                    self.gameList.pop()
                    return None
                self.gamePlayedList[index] += 1
                print("Score for thread", index, "at game", self.gamePlayedList[index],
                      "=", episode_params['score'])

            self.emit('get action',
                      {'thread_index': index,
                       'state': json.dumps(self.gameList[index].s_t, cls=NDArrayEncoder)})
        else:
            if episode_params['terminal']:
                self.gamePlayedList += 1
                print("Score for agent at game", self.gamePlayedList, "=", episode_params['score'])

            self.emit('get action',
                      {'state': json.dumps(self.gameList[0].s_t, cls=NDArrayEncoder)})

        key = self.inputKey.data
        if key != '':
            print(key)
            if key[-2] == "d":
                print('Playing game for training algorithm at current step...')
                self.gameList.append(Game(0, self.cfg.game_rom, display=True, no_op_max=0))
                self.play_thread = threading.Thread(
                    target=self.emit('get action',
                                     {'thread_index': -1,
                                      'state': json.dumps(self.gameList[-1].s_t, cls=NDArrayEncoder)}))
                self.play_thread.start()
            self.inputKey.isFinished = True
            self.inputKey = bgread(sys.stdin)

    def on_stop_training_ack(self, *args):
        print('on_stop_training_ack', args)
        self.emit('disconnect', {})

socketIO = SocketIO('localhost', 8000)
rlmodels_namespace = socketIO.define(ServerAPI, '/rlmodels')
socketIO.wait(seconds=1)
