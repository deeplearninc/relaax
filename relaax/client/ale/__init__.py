from __future__ import print_function

import argparse
import logging
import random

from . import game_process
from ...common.loop import socket_loop


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default=None, help='agent server address (host:port)')
    parser.add_argument('--rom', type=str, help='Atari game ROM file')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random generator')
    return parser.parse_args()


class _Environment(object):
    def __init__(self, game):
        self._game = game
        self._n_game = 0

    def get_state(self):
        return self._game.state()

    def act(self, action):
        return self._game.act(action)

    def reset(self, score):
        self._n_game += 1
        print('Score at game', self._n_game, '=', score)
        self._game.reset()


def main():
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s: %(message)s',
        level=logging.INFO
    )

    args = parse_args()
    socket_loop.run_environment(
        args.agent,
        _Environment(game_process.GameProcessFactory(args.rom).new_env(_seed(args.seed)))
    )


def _seed(value):
    if value is None:
        return random.randrange(1000000)
    return value
