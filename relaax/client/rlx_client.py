from __future__ import print_function

from ..common.protocol import socket_protocol


Failure = socket_protocol.Failure


class Client(object):
    def __init__(self, socket):
        self._socket = socket
        self._agent_service = socket_protocol.AgentStub(socket)

    def init(self, state):
        self._agent_service.act(state)
        return socket_protocol.environment_receive_act(self._socket)

    def send(self, reward, state):
        if reward is None:
            self._agent_service.act(state)
        else:
            self._agent_service.reward_and_act(reward, state)
        return socket_protocol.environment_receive_act(self._socket)

    def reset(self, reward):
        self._agent_service.reward_and_reset(reward)
        return socket_protocol.environment_receive_reset(self._socket)

    def disconnect(self):
        raise NotImplementedError
