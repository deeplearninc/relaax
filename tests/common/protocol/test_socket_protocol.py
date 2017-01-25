import base64
import io
import json
import mock
import numpy
import struct
import unittest

from relaax.common.protocol import socket_protocol


class TestSocketProtocol_AgentStub(unittest.TestCase):
    def setUp(self):
        self.socket = mock.Mock()
        self.stub = socket_protocol.AgentStub(self.socket)

    def tearDown(self):
        pass

    def test_init(self):
        pass

    def test_act_on_dict(self):
        self.stub.act({'key': 'value'})
        self.socket.sendall.assert_called_once_with(self._data('["act", "{\\"key\\": \\"value\\"}"]'))

    def test_act_on_array(self):
        self.stub.act([0, 1])
        self.socket.sendall.assert_called_once_with(self._data('["act", "[0, 1]"]'))

    def test_act_on_numpy_array(self):
        nparray = numpy.array([16, 17])
        self.stub.act(nparray)
        output = io.BytesIO()
        numpy.savez_compressed(output, obj=nparray)
        self.socket.sendall.assert_called_once_with(self._data(
            '["act", "{\\"b64npz\\": \\"%s\\"}"]' % base64.b64encode(output.getvalue())
        ))

    def test_reward_and_reset(self):
        self.stub.reward_and_reset(16)
        self.socket.sendall.assert_called_once_with(self._data('["reward_and_reset", 16]'))

    def test_reward_and_act_on_dict(self):
        self.stub.reward_and_act(17, {'key': 'value'})
        self.socket.sendall.assert_called_once_with(self._data('["reward_and_act", 17, "{\\"key\\": \\"value\\"}"]'))

    def test_reward_and_act_on_array(self):
        self.stub.reward_and_act(18, [0, 1])
        self.socket.sendall.assert_called_once_with(self._data('["reward_and_act", 18, "[0, 1]"]'))

    def test_reward_and_act_on_numpy_array(self):
        nparray = numpy.array([16, 17])
        self.stub.reward_and_act(19, nparray)
        output = io.BytesIO()
        numpy.savez_compressed(output, obj=nparray)
        self.socket.sendall.assert_called_once_with(self._data(
            '["reward_and_act", 19, "{\\"b64npz\\": \\"%s\\"}"]' % base64.b64encode(output.getvalue())
        ))

    def test_metrics(self):
        self.stub.metrics().scalar('key', 21)
        self.socket.mock_calls
        self.socket.sendall.assert_called_once_with(self._data('["scalar_metric", "key", 21]'))

    def _data(self, data):
        return ''.join([
            struct.pack('<I', len(data)),
            data
        ])



class _MockSocket(object):
    pass
