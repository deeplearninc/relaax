from __future__ import print_function

import argparse
import socketIO_client
import logging
import random

import game_process
from .. import server_api


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument("--game", type=str, default="boxing", help="Name of the Atari game ROM")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random generator")
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
            game_process.GameProcessFactory(args_.game).new_env(_seed(args_.seed)),
            *args,
            **kwargs
        )


def _seed(value):
    if value is None:
        return random.randrange(1000000)
    return value

if __name__ == "__main__":
    main()
