from __future__ import print_function

from socketIO_client import SocketIO
import logging

from time import sleep

from game_process import GameProcess as Game
from params import Params

from .. import server_api


def main():
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO)

    socketIO = SocketIO('localhost', 8000)
    rlmodels_namespace = socketIO.define(ServerAPI, '/rlmodels')
    socketIO.wait(seconds=1)


class ServerAPI(server_api.ServerAPI):
    def __init__(self, *args, **kwargs):
        server_api.ServerAPI.__init__(self, Params(), *args, **kwargs)

    def make_game(self, seed):
        return Game(seed, self.cfg.game_rom)

    def make_display_game(self, seed):
        return Game(seed, self.cfg.game_rom, display=True, no_op_max=0)

    def stop_play_thread(self):
        self.play_thread.join()
        sleep(3)
        self.gameList.pop()


if __name__ == "__main__":
    main()
