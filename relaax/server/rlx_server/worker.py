from __future__ import print_function

import logging
import time

from ...common.loop import socket_loop


class Worker(object):
    def __init__(self, agent_factory, timeout, n_agent, connection, address):
        self._agent_factory = agent_factory
        self._timeout = timeout
        self._n_agent = n_agent
        self._connection = connection
        self._address = address

    def run(self):
        try:
            socket_loop.run_agent(
                self._connection,
                self._address,
                _AgentServiceFactory(self._agent_factory(self._n_agent), self._timeout)
            )
        except _Failure as e:
            _warning('{} : {}'.format(self._address, e.message))


class _AgentService(socket_loop.AgentService):
    def __init__(self, environment_service, agent, timeout):
        self._stop = time.time() + timeout
        self._environment_service = environment_service
        self._agent = agent

    def act(self, state):
        self._environment_service.act(self._agent.act(state))

    def reward_and_reset(self, reward):
        score = self._agent.reward_and_reset(reward)
        if score is None:
            raise _Failure('no answer from agent')
        if time.time() >= self._stop:
            raise _Failure('timeout')
        self._environment_service.reset(score)

    def reward_and_act(self, reward, state):
        action = self._agent.reward_and_act(reward, state)
        if action is None:
            raise _Failure('no answer from agent')
        self._environment_service.act(action)


class _AgentServiceFactory(socket_loop.AgentServiceFactory):
    def __init__(self, agent, timeout):
        self._agent = agent
        self._timeout = timeout

    def __call__(self, environment_service):
        return _AgentService(environment_service, self._agent, self._timeout)


class _Failure(Exception):
    def __init__(self, message):
        self.message = message


def _warning(message, *args):
    logging.warning('%d:' + message, os.getpid(), *args)
