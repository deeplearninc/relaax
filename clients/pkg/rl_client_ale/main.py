from __future__ import print_function

import argparse
import socketIO_client
import logging
import time
import random

import game_process
import params
from .. import server_api


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument("--game", type=str, default="boxing", help="Name of the Atari game ROM")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random generator")
    parser.add_argument(
        "--lstm",
        type=lambda s: s.lower() not in ('no', 'n', 'false', 'f', '0', ''),
        default=True,
        help="Adds LSTM layer before net output"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO)

    socketIO = socketIO_client.SocketIO(args.host, args.port)
    rlmodels_namespace = socketIO.define(_ServerAPI, '/rlmodels')
    socketIO.wait(seconds=1)


class _ServerAPI(server_api.ServerAPI):
    def __init__(self, *args, **kwargs):
        args_ = parse_args()
        server_api.ServerAPI.__init__(
            self,
            _seed(args_.seed),
            game_process.GameProcessFactory(args_.game),
            *args,
            **kwargs
        )

    def stop_play_thread(self):
        self.play_thread.join()
        time.sleep(3)
        self.gameList.pop()


def _seed(value):
    if value is None:
        return random.randrange(1000000)
    return int(value)

if __name__ == "__main__":
    main()
