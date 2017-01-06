from __future__ import print_function

import threading

from .common.protocol import socket_protocol


class Client(object):
    def init(self, state):
        raise NotImplementedError

    def send(self, reward, state):
        raise NotImplementedError

    def reset(self, reward):
        raise NotImplementedError

    def disconnect(self):
        raise NotImplementedError


Failure = socket_protocol.Failure


class SyncSocketClient(Client):
    def __init__(self, socket):
        self._socket = socket
        self._agent_service = socket_protocol.AgentStub(socket)
        self._environment_service = _EnvironmentService()

    def init(self, state):
        self._agent_service.act(state)
        return self._on_act()

    def send(self, reward, state):
        if reward is None:
            self._agent_service.act(state)
        else:
            self._agent_service.reward_and_act(reward, state)
        return self._on_act()

    def reset(self, reward):
        self._agent_service.reward_and_reset(reward)
        return self._on_reset()

    def _on_act(self):
        socket_protocol.environment_dispatch(self._socket, self._environment_service)
        assert self._environment_service.action is not None
        return self._environment_service.action

    def _on_reset(self):
        socket_protocol.environment_dispatch(self._socket, self._environment_service)
        assert self._environment_service.episode_score is not None
        return self._environment_service.episode_score


class _EnvironmentService(socket_protocol.EnvironmentService):
    def __init__(self):
        self.action = None
        self.episode_score = None

    def act(self, action):
        self.action = action
        self.episode_score = None

    def reset(self, episode_score):
        self.action = None
        self.episode_score = episode_score
