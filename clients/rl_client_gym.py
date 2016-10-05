from __future__ import print_function

from socketIO_client import SocketIO
import logging

from time import sleep

from rl_client_gym.game_env import Env as Game
from rl_client_gym.params import Params

import server_api


class ServerAPI(server_api.ServerAPI):
    def __init__(self, *args, **kwargs):
        server_api.ServerAPI.__init__(self, Params(), *args, **kwargs)

    def make_game(self, seed):
        return Game(self.cfg, seed)

    def make_display_game(self, seed):
        return Game(self.cfg, seed, display=True)

    def action_size(self):
        return self.gameList[0].getActions()  

    def game_state(self, i):
        return self.gameList[i].state

    def act(self, i, action):
        return self.gameList[i].act(action)

    def stop_play_thread(self):
        sleep(3)
        self.gameList[-1].gym.render(close=True)
        self.play_thread.join()
        self.gameList.pop()


if __name__ == "__main__":
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO)

    socketIO = SocketIO('localhost', 8000)
    rlmodels_namespace = socketIO.define(ServerAPI, '/rlmodels')
    socketIO.wait(seconds=1)
