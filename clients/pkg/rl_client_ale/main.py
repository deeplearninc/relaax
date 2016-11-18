from __future__ import print_function

import sys
sys.path.append('../../pkg')

import argparse
import logging
import random

import game_process
import loop.socket_loop


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default=None, help='agent server address (host:port)')
    parser.add_argument("--game", type=str, default="boxing", help="Name of the Atari game ROM")
    parser.add_argument("--seed", type=int, default=None, help="Seed for random generator")
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
        print("Score at game", self._n_game, "=", score)
        self._game.reset()


def main():
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s: %(message)s',
        level=logging.INFO
    )

    args = parse_args()
    loop.socket_loop.run_environment(
        args.agent,
        _Environment(game_process.GameProcessFactory(args.game).new_env(_seed(args.seed)))
    )


def _seed(value):
    if value is None:
        return random.randrange(1000000)
    return value


if __name__ == "__main__":
    main()
