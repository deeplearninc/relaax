from __future__ import print_function

import socket

from ..common.protocol import socket_protocol


Failure = socket_protocol.Failure


class Client(object):
    def __init__(self, rlx_server_url):
        self._socket = None
        self._agent_service = None

        try:
            self._socket = socket.socket()
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._socket.connect(_parse_address(rlx_server_url))
        except socket.error as e:
            if self._socket is not None:
                self._socket.close()
            raise Failure("socket error({}): {}".format(e.errno, e.strerror))
        assert self._socket is not None

        try:
            self._agent_service = socket_protocol.AgentStub(self._socket)
        except:
            self._socket.close()
            raise
        assert self._agent_service is not None

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

    def store_scalar_metric(self, name, y, x=None):
        self._agent_service.store_scalar_metric(name, y, x)

    def disconnect(self):
        self._socket.close()


def _parse_address(address):
    host, port = address.split(':')
    return host, int(port)
