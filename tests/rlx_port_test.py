from __future__ import print_function
from __future__ import absolute_import
from builtins import str
from builtins import object
import errno
import os
import socket
import signal
import traceback
from mock import Mock
import sys

from .fixtures.mock_cmdl import cmdl
from .fixtures.mock_utils import MockUtils
from .fixtures.mock_socket import MockSocket

from relaax.server.rlx_server.rlx_port import RLXPort


class TestRLXPort(object):

    @classmethod
    def teardown_class(cls):
        cmdl.restore()

    def setup_method(self, method):
        self.socket = MockSocket.create()

    def test_socket_error_on_accept(self, monkeypatch):
        monkeypatch.setattr(socket, 'socket', lambda af, st: self.socket)
        self.socket.accept = lambda: MockUtils.raise_(
            socket.error(errno.ECONNABORTED, "error message"))
        try:
            RLXPort.listen(('localhost', 7000))
            assert False
        except Exception as e:
            assert str(e) == '[Errno %d] error message' % errno.ECONNABORTED

    def test_socket_error_on_accept2(self, monkeypatch):
        def error(*args):
            called.times += 1
            called.args = args

        logger = Mock()
        called = MockUtils.Placeholder()
        logger.error = error
        monkeypatch.setattr('relaax.server.rlx_server.rlx_port.log', logger)
        monkeypatch.setattr(socket, 'socket', lambda af, st: self.socket)
        self.socket.accept = lambda: MockUtils.raise_(socket.error(errno.ENOMEM, "fatal error message"))
        try:
            RLXPort.listen(('localhost', 7000))
            assert False
        except Exception as e:
            assert called.args == ('Could not accept new connection (fatal error message)',)
            assert called.times == 1
            assert str(e) == '[Errno %d] fatal error message' % errno.ENOMEM

    def test_socket_busy_on_accept(self, monkeypatch):
        accept_responses = [
            lambda: MockUtils.raise_(socket.error(errno.ENOBUFS, "fatal error message")),
            lambda: MockUtils.raise_(socket.error(errno.EAGAIN, "busy, try accept again")),
            lambda: MockUtils.raise_(socket.error(errno.EPERM, "rejected, but try accept again"))
        ]
        monkeypatch.setattr(socket, 'socket', lambda af, st: self.socket)
        self.socket.accept = lambda: accept_responses.pop()()
        try:
            RLXPort.listen(('localhost', 7000))
            assert False
        except Exception as e:
            traceback.format_exc()
            assert str(e) == '[Errno %d] fatal error message' % errno.ENOBUFS

    def test_keyboard_interrupt_on_accept(self, monkeypatch):
        if sys.platform != 'win32':
            monkeypatch.setattr(os, 'fork', lambda: 0)
            monkeypatch.setattr(socket, 'socket', lambda af, st: self.socket)
            self.socket.accept = lambda: os.kill(os.getpid(), signal.SIGINT)
            RLXPort.listen(('localhost', 7000))
