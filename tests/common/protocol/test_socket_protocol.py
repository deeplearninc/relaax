import base64
import io
import json
import mock
import numpy
import struct
import unittest

import relaax.algorithm_base.agent_base
import relaax.common.metrics

from relaax.common.protocol import socket_protocol


class TestSocketProtocol_agent(unittest.TestCase):
    def setUp(self):
        self.socket = _MockSocket()
        self.stub = socket_protocol.AgentStub(self.socket)
        self.calls = []
        self.service = _MockService(self.calls)

    def tearDown(self):
        pass

    def test_act(self):
        state = {
            'boolean': True,
            'int': 16,
            'float': 1.5,
            'string': 's',
            'array': [17, 18],
            'nparray': numpy.array([19, 20])
        }
        self.stub.act(state)
        socket_protocol.agent_dispatch(self.socket, self.service)
        self.assertTrue(eq([('act', state)], self.calls))

    def test_reward_and_reset(self):
        self.stub.reward_and_reset(16.1)
        socket_protocol.agent_dispatch(self.socket, self.service)
        self.assertTrue(eq([('reward_and_reset', 16.1)], self.calls))

    def test_reward_and_act(self):
        state = {
            'boolean': False,
            'int': 116,
            'float': 2.5,
            'string': 'sS',
            'array': [117, 118],
            'nparray': numpy.array([119, 120])
        }
        self.stub.reward_and_act(115.1, state)
        socket_protocol.agent_dispatch(self.socket, self.service)
        self.assertTrue(eq([('reward_and_act', 115.1, state)], self.calls))

    def test_metrics(self):
        self.stub.metrics().scalar('key', 21.2)
        socket_protocol.agent_dispatch(self.socket, self.service)
        self.stub.metrics().scalar('key', 21.3, x=17.1)
        socket_protocol.agent_dispatch(self.socket, self.service)
        self.assertTrue(eq([
            ('scalar_metric', 'key', 21.2, None),
            ('scalar_metric', 'key', 21.3, 17.1)
        ], self.calls))

    def test_loop(self):
        state1 = numpy.array([3.1, 4.2])
        self.stub.act(state1)
        socket_protocol.agent_dispatch(self.socket, self.service)

        self.stub.metrics().scalar('reward', 21.2)
        socket_protocol.agent_dispatch(self.socket, self.service)

        state2 = numpy.array([4.1, 5.2])
        self.stub.reward_and_act(115.1, state2)
        socket_protocol.agent_dispatch(self.socket, self.service)

        self.stub.metrics().scalar('reward', 22.2, x=11.1)
        socket_protocol.agent_dispatch(self.socket, self.service)

        state3 = numpy.array([5.1, 6.2])
        self.stub.reward_and_act(116.1, state3)
        socket_protocol.agent_dispatch(self.socket, self.service)

        self.stub.metrics().scalar('reward', 23.2)
        socket_protocol.agent_dispatch(self.socket, self.service)

        self.stub.reward_and_reset(117.1)
        socket_protocol.agent_dispatch(self.socket, self.service)

        self.stub.metrics().scalar('reward', 24.2)
        socket_protocol.agent_dispatch(self.socket, self.service)

        state4 = numpy.array([6.1, 7.2])
        self.stub.reward_and_act(118.1, state4)
        socket_protocol.agent_dispatch(self.socket, self.service)

        self.assertTrue(eq([
            ('act'             , state1              ),
            ('scalar_metric'   , 'reward', 21.2, None),
            ('reward_and_act'  , 115.1, state2       ),
            ('scalar_metric'   , 'reward', 22.2, 11.1),
            ('reward_and_act'  , 116.1, state3       ),
            ('scalar_metric'   , 'reward', 23.2, None),
            ('reward_and_reset', 117.1               ),
            ('scalar_metric'   , 'reward', 24.2, None),
            ('reward_and_act'  , 118.1, state4       )
        ], self.calls))


class TestSocketProtocol_environment(unittest.TestCase):
    def setUp(self):
        self.socket = _MockSocket()

    def tearDown(self):
        pass

    def test_act(self):
        action = {
            'boolean': False,
            'int': 116,
            'float': 2.5,
            'string': 'sS',
            'array': [117, 118],
            'nparray': numpy.array([119, 120])
        }
        socket_protocol.environment_send_act(self.socket, action)
        self.assertTrue(eq(
            action,
            socket_protocol.environment_receive_act(self.socket)
        ))

    def test_reset(self):
        socket_protocol.environment_send_reset(self.socket, 14.1)
        self.assertEquals(
            14.1,
            socket_protocol.environment_receive_reset(self.socket)
        )


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
    def __init__(self, calls):
        self._calls = calls
        self._metrics = _MockMetrics(calls, 'scalar_metric')

    def act(self, state):
        self._calls.append(('act', state))

    def reward_and_reset(self, reward):
        self._calls.append(('reward_and_reset', reward))

    def reward_and_act(self, reward, state):
        self._calls.append(('reward_and_act', reward, state))

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
