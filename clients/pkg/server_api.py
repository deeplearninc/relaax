from __future__ import print_function

import base64
import io
import json
import numpy
import sys
import socketIO_client


class _NDArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            output = io.BytesIO()
            numpy.savez_compressed(output, obj=obj)
            return {'b64npz': base64.b64encode(output.getvalue())}
        return json.JSONEncoder.default(self, obj)


class ServerAPI(socketIO_client.LoggingNamespace):
    def __init__(self, game, *args, **kwargs):
        socketIO_client.LoggingNamespace.__init__(self, *args, **kwargs)
        self._game = game
        self._game_played = 0

    def on_connected(self, *args):
        print('on_connected')
        self.emit('get action', {'state': self._dump_state()})

    def on_get_action_ack(self, *args):
        action = args[0]['action']

        reward = self._game.act(action)
        terminal = self._game.terminal

        if terminal:
            self._game.reset()

        self.emit('episode', {
            'reward': reward,
            'terminal': terminal
        })

    def on_episode_ack(self, *args):
        episode_params = json.loads(args[0])
        if episode_params['stop_training']:
            self.emit('stop training')
            return

        if episode_params['terminal']:
            self._game_played += 1
            print("Score at game", self._game_played, "=", int(episode_params['score']))

        self.emit('get action', {'state': self._dump_state()})

    def on_stop_training_ack(self, *args):
        print('on_stop_training_ack', args)
        self.emit('disconnect', {})

    def _dump_state(self):
        return json.dumps(self._game.state(), cls=_NDArrayEncoder)
