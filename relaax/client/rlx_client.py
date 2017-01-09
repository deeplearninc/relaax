from __future__ import print_function

import socket

from ..common.protocol import socket_protocol


Failure = socket_protocol.Failure


class Client(object):
    def __init__(self, rlx_server):
        self._host, self._port = _parse_address(rlx_server)
        self._socket = None

    def __enter__(self):
        self._socket = socket.socket()
        try:
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._socket.connect((self._host, self._port))
        except socket.error as e:
            self._socket.close()
            raise Failure("socket error({}): {}".format(e.errno, e.strerror))
        return _Client(self._socket)

    def __exit__(self, exc_type, exc_value, traceback):
        if self._socket is not None:
            self._socket.close()


class _Client(object):
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

    def store_scalar_metric(self, name, y, x=None):
        self._agent_service.store_scalar_metric(name, y, x)

    def disconnect(self):
        raise NotImplementedError


def _parse_address(address):
    host, port = address.split(':')
    return host, int(port)
