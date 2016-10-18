from __future__ import print_function

import argparse

from socketIO_client import SocketIO
import logging

from time import sleep

import game_process
from params import Params

from .. import server_api


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument("--scope", type=str, default="ale_model", help="Name of model scope")
    parser.add_argument("--algo", type=str, default="a3c", help="Name of the RL algorithm to perform")
    parser.add_argument("--game", type=str, default="boxing", help="Name of the Atari game ROM")
    parser.add_argument("--agents", type=int, default=8, help="Number of parallel training Agents")
    parser.add_argument("--lstm", type=bool, default=True, help="Adds LSTM layer before net output")
    return parser.parse_args()


def main():
    args = parse_args()

    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO)

    socketIO = SocketIO(args.host, args.port)
    rlmodels_namespace = socketIO.define(ServerAPI, '/rlmodels')
    socketIO.wait(seconds=1)


class ServerAPI(server_api.ServerAPI):
    def __init__(self, *args, **kwargs):
        params = Params(parse_args())
        server_api.ServerAPI.__init__(
            self,
            params,
            game_process.GameProcessFactory(params),
            *args,
            **kwargs
        )

    def stop_play_thread(self):
        self.play_thread.join()
        sleep(3)
        self.gameList.pop()


if __name__ == "__main__":
    main()
