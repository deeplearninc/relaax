from __future__ import print_function

import logging
import os
import time

import relaax.algorithm_base.agent_base
from ...common.protocol import socket_protocol


class Worker(object):
    def __init__(
        self,
        agent_factory,
        timeout,
        n_agent,
        connection,
        address,
        agent_service_factory=lambda connection, agent, timeout:
            _AgentService(connection, agent, timeout)
    ):
        self._connection = connection
        self._address = address
        self._agent_service_factory = lambda: agent_service_factory(
            connection=connection,
            agent=agent_factory(n_agent),
            timeout=timeout
        )

    def run(self):
        agent_service = self._agent_service_factory()
        try:
            while True:
                socket_protocol.agent_dispatch(self._connection, agent_service)
        except socket_protocol.Failure as e:
            logging.warning('{}: {}: {}'.format(os.getpid(), self._address, e.message))


class _AgentService(relaax.algorithm_base.agent_base.AgentBase):
    def __init__(self, connection, agent, timeout):
        self._connection = connection
        self._agent = agent
        self._stop = time.time() + timeout

    def act(self, state):
        socket_protocol.environment_send_act(self._connection, self._agent.act(state))

    def reward_and_reset(self, reward):
        score = self._agent.reward_and_reset(reward)
        if score is None:
            raise socket_protocol.Failure('no answer from agent')
        socket_protocol.environment_send_reset(self._connection, score)
        if time.time() >= self._stop:
            raise socket_protocol.Failure('timeout')

    def reward_and_act(self, reward, state):
        action = self._agent.reward_and_act(reward, state)
        if action is None:
            raise socket_protocol.Failure('no answer from agent')
        socket_protocol.environment_send_act(self._connection, action)

    def metrics(self):
        return self._agent.metrics()
