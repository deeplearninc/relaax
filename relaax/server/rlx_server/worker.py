from __future__ import print_function

import logging
import os
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
        agent_service = _AgentService(
            environment_service=socket_loop.EnvironmentStub(self._connection),
            agent=self._agent_factory(self._n_agent),
            timeout=self._timeout
        )
        try:
            while True:
                socket_loop.agent_dispatch(self._connection, agent_service)
        except socket_loop.Failure as e:
            logging.warning('{}: {}: {}'.format(os.getpid(), self._address, e.message))


class _AgentService(socket_loop.AgentService):
    def __init__(self, environment_service, agent, timeout):
        self._environment_service = environment_service
        self._agent = agent
        self._stop = time.time() + timeout

    def act(self, state):
        self._environment_service.act(self._agent.act(state))

    def reward_and_reset(self, reward):
        score = self._agent.reward_and_reset(reward)
        if score is None:
            raise socket_loop.Failure('no answer from agent')
        if time.time() >= self._stop:
            raise socket_loop.Failure('timeout')
        self._environment_service.reset(score)

    def reward_and_act(self, reward, state):
        action = self._agent.reward_and_act(reward, state)
        if action is None:
            raise socket_loop.Failure('no answer from agent')
        self._environment_service.act(action)


