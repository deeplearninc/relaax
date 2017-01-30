from __future__ import print_function

import socket

from ..common.protocol import socket_protocol


Failure = socket_protocol.Failure


def _socket_factory(rlx_server_url):
    try:
        s = socket.socket()
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.connect(_parse_address(rlx_server_url))
    except socket.error as e:
        if s is not None:
            s.close()
        raise Failure("socket error({}): {}".format(e.errno, e.strerror))
    assert s is not None
    return s


class Client(object):
    def __init__(self, rlx_server_url, socket_factory=_socket_factory):
        self._socket = socket_factory(rlx_server_url)
        self._agent_service = None

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

    def metrics(self):
        return self._agent_service.metrics()

    def disconnect(self):
        self._socket.close()


def _parse_address(address):
    host, port = address.split(':')
    return host, int(port)
