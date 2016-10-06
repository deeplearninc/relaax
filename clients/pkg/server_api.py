from __future__ import print_function

import base64
import io
import json
import nonblock
import numpy
import sys
import threading
import socketIO_client


class NDArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            output = io.BytesIO()
            numpy.savez_compressed(output, obj=obj)
            return {'b64npz': base64.b64encode(output.getvalue())}
        return json.JSONEncoder.default(self, obj)


class ServerAPI(socketIO_client.LoggingNamespace):
    def __init__(self, cfg, *args, **kwargs):
        socketIO_client.LoggingNamespace.__init__(self, *args, **kwargs)

        self.inputKey = nonblock.bgread(sys.stdin)   # for Display --> standard console input interception
        self.play_thread = None     # Display thread --> need for further deleting
        self.gamePlayedList = None  # Games played accumulator for current session (can holds all threads)
        self.cfg = cfg              # Parameters to setup the environment and training process
        self.gameList = []          # List of running games, if no parallel agents -> list holds one element

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
                self.gameList.append(self.make_game(113 * i))
        else:
            self.gameList.append(self.make_game(0))
        self.cfg.action_size = self.action_size()

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
            self.gamePlayedList = numpy.zeros(args[0]['threads_cnt'], dtype=int)

            for i in range(args[0]['threads_cnt']):
                print('Game agent\'s thread is created with index:', i)
                threading.Thread(target=self.emit('get action', {
                    'thread_index': i,
                    'state': self.dump_state(i)
                }))
        else:
            self.gamePlayedList = 0
            print('Game is created for the training...')
            self.emit('get action', {
                'state': self.dump_state(0)
            })

    def on_get_action_ack(self, *args):
        # if more than one agent trains -> server returns tuple(list) with [action_num, thread_num]
        if hasattr(args[0]['action'], '__getitem__'):
            action = args[0]['action'][0]
            index = args[0]['action'][1]
        else:
            action = args[0]['action']
            index = 0

        # receive game result
        reward = self.act(index, action)
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
                self.gameList.append(self.make_display_game(0))
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

    def make_game(self, seed):
        raise NotImplementedError()

    def make_display_game(self, seed):
        raise NotImplementedError()

    def action_size(self):
        raise NotImplementedError()

    def game_state(self, i):
        raise NotImplementedError()

    def act(self, i, action):
        raise NotImplementedError()

    def dump_state(self, i):
        return json.dumps(self.game_state(i), cls=NDArrayEncoder)
