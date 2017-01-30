import io
import logging
import mock
import numpy
import unittest

from relaax.common.protocol import socket_protocol

from relaax.server.rlx_server import worker

class TestWorker(unittest.TestCase):
    def setUp(self):
        self.connection = _Socket()
        self.agent = mock.Mock()
        self.worker = worker.Worker(
            agent_factory=lambda n_agent: self.agent, # is not used in testing
            timeout=1000, # arbitrary large number
            n_agent=0, # arbitrary agent number
            connection=self.connection,
            address=None
        )
        self.stub = socket_protocol.AgentStub(self.connection.input)
        self.logger = logging.getLogger(worker.__name__)
        self.old_level = self.logger.level
        self.logger.setLevel(logging.ERROR)

    def tearDown(self):
        self.logger.setLevel(self.old_level)

    def test_run_on_act(self):
        state = numpy.array([19.1, 20.2])
        action = numpy.array([1.1, 2.2])
        self.agent.act = mock.Mock(return_value=action)
        self.stub.act(state)

        self.worker.run()

        self.assertEquals(1, len(self.agent.act.mock_calls))
        self.assertTrue((self.agent.act.mock_calls[0][1] == state).all())
        self.assertTrue((
            action == socket_protocol.environment_receive_act(self.connection.output)
        ).all())

    def test_run_on_reward_and_reset(self):
        self.agent.reward_and_reset = mock.Mock(return_value=4.3)
        self.stub.reward_and_reset(3.4)

        self.worker.run()

        self.assertEquals(1, len(self.agent.reward_and_reset.mock_calls))
        self.assertEquals((3.4,), self.agent.reward_and_reset.mock_calls[0][1])
        self.assertTrue(4.3, socket_protocol.environment_receive_reset(self.connection.output))

    def test_run_on_reward_and_act(self):
        state = numpy.array([19.1, 20.2])
        action = numpy.array([1.1, 2.2])
        self.agent.reward_and_act = mock.Mock(return_value=action)
        self.stub.reward_and_act(5.6, state)

        self.worker.run()

        self.assertEquals(1, len(self.agent.reward_and_act.mock_calls))
        self.assertEquals(2, len(self.agent.reward_and_act.mock_calls[0][1]))
        self.assertEquals(5.6, self.agent.reward_and_act.mock_calls[0][1][0])
        self.assertTrue((self.agent.reward_and_act.mock_calls[0][1][1] == state).all())
        self.assertTrue((
            action == socket_protocol.environment_receive_act(self.connection.output)
        ).all())

    def test_run_on_metrics(self):
        scalar_mock = mock.Mock(return_value=4.3)
        self.agent.metrics = mock.Mock(return_value=scalar_mock)
        self.stub.metrics().scalar('key', 3.4, x=4.5)
        self.stub.metrics().scalar('key', 6.5)

        self.worker.run()

        self.assertEquals([
            mock.call.scalar('key', 3.4, 4.5),
            mock.call.scalar('key', 6.5     )
        ], scalar_mock.mock_calls)


class _Socket(object):
    def __init__(self):
        self.output = _MockSocket()
        self.input = _MockSocket()
        self.opened = True

    def sendall(self, data):
        assert self.opened
        self.output.sendall(data)

    def recv(self, n):
        assert self.opened
        return self.input.recv(n)

    def close(self):
        assert self.opened
        self.opened = False


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


