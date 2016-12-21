from __future__ import print_function

import argparse
import logging
import random

from . import game_process
from ...common.loop import socket_loop


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rlx-server', type=str, default=None, help='RLX server address (host:port)')
    parser.add_argument('--ale', type=str, help='path to Arcade-Learning-Environment directory')
    parser.add_argument('--rom', type=str, help='Atari game ROM file')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random generator')
    return parser.parse_args()


class _EnvironmentService(socket_loop.EnvironmentService):
    def __init__(self, agent_service, environment):
        self._agent_service = agent_service
        self._environment = environment
        agent_service.act(environment.get_state())

    def act(self, action):
        reward, reset = self._environment.act(action)
        if reset:
            self._agent_service.reward_and_reset(reward)
        else:
            self._agent_service.reward_and_act(reward, self._environment.get_state())

    def reset(self, episode_score):
        self._environment.reset(episode_score)
        self._agent_service.act(self._environment.get_state())


class _EnvironmentServiceFactory(socket_loop.EnvironmentServiceFactory):
    def __init__(self, environment):
        self._environment = environment

    def __call__(self, agent_service):
        return _EnvironmentService(agent_service, self._environment)


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
        args.rlx_server,
        _EnvironmentServiceFactory(
            _Environment(game_process.GameProcessFactory(args.ale, args.rom).new_env(_seed(args.seed)))
        )
    )


def _seed(value):
    if value is None:
        return random.randrange(1000000)
    return value
