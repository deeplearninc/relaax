import io
import mock
import numpy
import unittest

import relaax.algorithm_base.agent_base

from relaax.client import rlx_client
from relaax.common.protocol import socket_protocol


class TestRlxClient(unittest.TestCase):
    def setUp(self):
        self.socket = _Socket()
        self.client = rlx_client.Client(None, lambda rlx_server_url: self.socket)
        self.service = _MockService()

    def tearDown(self):
        pass

    def test_init(self):
        state = numpy.array([6.1, 7.2, 8.3])
        action = numpy.array([9.4, 10.5])
        socket_protocol.environment_send_act(self.socket.input, action)

        self.assertTrue((action == self.client.init(state)).all())

        socket_protocol.agent_dispatch(self.socket.output, self.service)

        self.assertTrue(eq([('act', state)], self.service.calls))


    def test_send(self):
        state = numpy.array([6.1, 7.2, 8.3])
        action = numpy.array([9.4, 10.5])
        socket_protocol.environment_send_act(self.socket.input, action)

        self.assertTrue((action == self.client.send(11.3, state)).all())

        socket_protocol.agent_dispatch(self.socket.output, self.service)

        self.assertTrue(eq([('reward_and_act', 11.3, state)], self.service.calls))

    def test_send_none(self):
        state = numpy.array([6.1, 7.2, 8.3])
        action = numpy.array([9.4, 10.5])
        socket_protocol.environment_send_act(self.socket.input, action)

        self.assertTrue((action == self.client.send(None, state)).all())

        socket_protocol.agent_dispatch(self.socket.output, self.service)

        self.assertTrue(eq([('act', state)], self.service.calls))

    def test_reset(self):
        socket_protocol.environment_send_reset(self.socket.input, 16.3)

        self.assertEquals(16.3, self.client.reset(11.3))

        socket_protocol.agent_dispatch(self.socket.output, self.service)

        self.assertTrue(eq([('reward_and_reset', 11.3)], self.service.calls))

    def test_metrics(self):
        self.client.metrics().scalar('name'        , 11.3, x=31.1)
        self.client.metrics().scalar('another name', 11.4        )

        socket_protocol.agent_dispatch(self.socket.output, self.service)
        socket_protocol.agent_dispatch(self.socket.output, self.service)

        self.assertTrue(eq([
            ('scalar_metric', 'name'        , 11.3, 31.1),
            ('scalar_metric', 'another name', 11.4, None)
        ], self.service.calls))

def _socket_factory(rlx_server_url):
    socket = mock.Mock()
    return _Socket


class _Socket(object):
    def __init__(self):
        self.output = _MockSocket()
        self.input = _MockSocket()

    def sendall(self, data):
        self.output.sendall(data)

    def recv(self, n):
        return self.input.recv(n)


class _MockSocket(object):
    def __init__(self):
        self._buffer = io.BytesIO()
        self._read_pos = self._buffer.tell()

    def sendall(self, data):
        self._buffer.seek(0, io.SEEK_END)
        self._buffer.write(data)

    def recv(self, n):
        self._buffer.seek(self._read_pos, io.SEEK_SET)
        bs = self._buffer.read(n)
        self._read_pos = self._buffer.tell()
        return bs


class _MockService(relaax.algorithm_base.agent_base.AgentBase):
    def __init__(self):
        self.calls = []
        self._metrics = _MockMetrics(self.calls, 'scalar_metric')

    def act(self, state):
        self.calls.append(('act', state))

    def reward_and_reset(self, reward):
        self.calls.append(('reward_and_reset', reward))

    def reward_and_act(self, reward, state):
        self.calls.append(('reward_and_act', reward, state))

    def metrics(self):
        return self._metrics


class _MockMetrics(relaax.common.metrics.Metrics):
    def __init__(self, calls, verb):
        self._calls = calls
        self._verb = verb

    def scalar(self, name, y, x=None):
        self._calls.append((self._verb, name, y, x))


def eq(a, b):
    if isinstance(a, numpy.ndarray):
        return isinstance(b, numpy.ndarray) and (a == b).all()
    if isinstance(a, dict):
        if isinstance(b, dict):
            if set(a.keys()) != set(b.keys()):
                return False
            for k in a.keys():
                if not eq(a[k], b[k]):
                    return False
            return True
        return False
    if isinstance(a, tuple):
        if isinstance(b, tuple):
            if len(a) != len(b):
                return False
            for i in xrange(len(a)):
                if not eq(a[i], b[i]):
                    return False
            return True
        return False
    if isinstance(a, list):
        if isinstance(b, list):
            if len(a) != len(b):
                return False
            for i in xrange(len(a)):
                if not eq(a[i], b[i]):
                    return False
            return True
        return False
    return a == b
