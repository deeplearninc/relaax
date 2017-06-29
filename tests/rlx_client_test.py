from __future__ import absolute_import
from builtins import str
from builtins import object
import errno
import time
import socket

from .fixtures.mock_utils import MockUtils
from .fixtures.mock_socket import MockSocket
from relaax.common.rlx_netstring import NetString
from relaax.environment.agent_proxy import AgentProxy, AgentProxyException


class TestAgentProxy(object):

    def test_connect(self, monkeypatch):
        skt = MockSocket.create()
        monkeypatch.setattr(socket, 'socket', lambda: skt)
        client = AgentProxy('localhost:7000')
        client.connect()
        assert client.skt == skt

    def test_connect_with_address_as_tuple(self, monkeypatch):
        skt = MockSocket.create()
        monkeypatch.setattr(socket, 'socket', lambda: skt)
        client = AgentProxy(('localhost', 7000))
        client.connect()
        assert client.skt == skt

    def test_wrong_address(self):
        try:
            AgentProxy('localhost:abc').connect()
            assert False
        except AgentProxyException as e:
            assert str(e) == 'can\'t parse server address.'

    def test_socket_error(self, monkeypatch):
        monkeypatch.setattr(time, 'sleep', lambda x: x)
        skt = MockSocket.create()
        skt.connect = lambda: socket.error(errno.ECONNABORTED)
        try:
            AgentProxy('localhost:7000').connect()
            assert False
        except AgentProxyException as e:
            assert str(e) == '[Errno %d] Connection refused' % errno.ECONNREFUSED

    def test_some_unknown_exception_in_netstring_constructor(self, monkeypatch):
        # paranoid a bit
        skt = MockSocket.create()
        monkeypatch.setattr(socket, 'socket', lambda: skt)
        monkeypatch.setattr(
            NetString,
            '__init__',
            lambda x, y: MockUtils.raise_(AgentProxyException('unknown bug')))
        try:
            AgentProxy('localhost:7000').connect()
            assert False
        except AgentProxyException as e:
            assert str(e) == 'unknown bug'

    def test_disconnect(self, monkeypatch):
        skt = MockSocket.create()
        monkeypatch.setattr(socket, 'socket', lambda: skt)
        client = AgentProxy('localhost:7000')
        client.connect()
        client.disconnect()
        assert client.skt is None

    def _mock_exchange(self, monkeypatch):
        called_with = [None]

        def exchange(*args):
            called_with[0] = args[1]
        monkeypatch.setattr(AgentProxy, '_exchange', exchange)
        return called_with

    def test_init(self, monkeypatch):
        called_with = self._mock_exchange(monkeypatch)
        c = AgentProxy('localhost:7000')
        c.init()
        assert called_with[0] == {'command': 'init', 'exploit': False}

    def test_update(self, monkeypatch):
        called_with = self._mock_exchange(monkeypatch)
        c = AgentProxy('localhost:7000')
        c.update(1, 2, True)
        assert called_with[0] == {
            'terminal': True, 'state': 2, 'reward': 1, 'command': 'update'}

    def test_reset(self, monkeypatch):
        called_with = self._mock_exchange(monkeypatch)
        c = AgentProxy('localhost:7000')
        c.reset()
        assert called_with[0] == {'command': 'reset'}

    def test_exchange(self, monkeypatch):
        skt = MockSocket.create()
        monkeypatch.setattr(socket, 'socket', lambda: skt)
        client = AgentProxy('localhost:7000')
        client.connect()
        ret = client._exchange({'response': 'action', 'data': [1, 2, 3]})
        assert ret == [1, 2, 3]

    def test_not_connected_exchange(self):
        client = AgentProxy('localhost:7000')
        try:
            client._exchange({'some': 'data'})
            assert False
        except AgentProxyException as e:
            assert str(e) == 'no connection is available.'

    def test_wrong_protocol_response(self, monkeypatch):
        skt = MockSocket.create()
        monkeypatch.setattr(socket, 'socket', lambda: skt)
        client = AgentProxy('localhost:7000')
        client.connect()
        try:
            client._exchange({'wrong': 'response'})
            assert False
        except AgentProxyException as e:
            assert str(e) == 'wring message format'

    def test_error_response(self, monkeypatch):
        skt = MockSocket.create()
        monkeypatch.setattr(socket, 'socket', lambda: skt)
        client = AgentProxy('localhost:7000')
        client.connect()
        try:
            client._exchange({'response': 'error'})
            assert False
        except AgentProxyException as e:
            assert str(e) == 'unknown error'
        try:
            client._exchange({'response': 'error', 'message': 'some error'})
            assert False
        except AgentProxyException as e:
            assert str(e) == 'some error'

    def test_low_level_exception_on_exchange(self, monkeypatch):
        transport = NetString('skt')
        transport.write_string = lambda x: MockUtils.raise_(Exception('some error'))
        client = AgentProxy('localhost:7000')
        client.transport = transport
        client.skt = 'skt'
        try:
            client._exchange({'some': 'data'})
            assert False
        except AgentProxyException as e:
            assert str(e) == 'some error'
