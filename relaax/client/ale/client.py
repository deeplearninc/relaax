from __future__ import print_function

import random

from . import game_process
from ...common.loop import socket_loop


def run(rlx_server, ale, rom, seed):
    socket_loop.run_environment(
        server_address=rlx_server,
        environment_service_factory=_EnvironmentServiceFactory(
            _Environment(game_process.GameProcessFactory(ale, rom).new_env(_seed(seed)))
        )
    )


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


def _seed(value):
    if value is None:
        return random.randrange(1000000)
    return value
