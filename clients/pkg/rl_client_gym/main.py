from __future__ import print_function

from socketIO_client import SocketIO
import logging

from time import sleep

import game_env
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
        params = Params()
        server_api.ServerAPI.__init__(
            self,
            params,
            game_env.EnvFactory(params),
            *args,
            **kwargs
        )

    def stop_play_thread(self):
        sleep(3)
        self.gameList[-1].gym.render(close=True)
        self.play_thread.join()
        self.gameList.pop()


if __name__ == "__main__":
    main()
