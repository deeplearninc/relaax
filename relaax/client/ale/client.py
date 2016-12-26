from __future__ import print_function

import logging
import os
import random
import socket
import time

from . import game_process
from ...common.protocol import socket_protocol


def run(rlx_server, ale, rom, seed):
    server_address=rlx_server
    environment = _Environment(
        game_process.GameProcessFactory(ale, rom).new_env(_seed(seed))
    )

    while True:
        s = socket.socket()
        try:
            try:
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                _connectf(s, _parse_address(server_address))
                environment_service = _EnvironmentService(
                    agent_service=socket_protocol.AgentStub(s),
                    environment=environment
                )
                while True:
                    socket_protocol.environment_dispatch(s, environment_service)
            finally:
                s.close()
        except socket_protocol.Failure as e:
            _warning('{} : {}'.format(server_address, e.message))
            delay = random.randint(1, 10)
            _info('waiting for %ds...', delay)
            time.sleep(delay)


class _EnvironmentService(socket_protocol.EnvironmentService):
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


def _parse_address(address):
    host, port = address.split(':')
    return host, int(port)


def _connectf(s, server_address):
    try:
        s.connect(server_address)
    except socket.error as e:
        raise socket_protocol.Failure("socket error({}): {}".format(e.errno, e.strerror))


def _info(message, *args):
    logging.info('%d:' + message, os.getpid(), *args)


def _warning(message, *args):
    logging.warning('%d:' + message, os.getpid(), *args)
