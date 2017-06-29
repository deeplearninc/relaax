from __future__ import absolute_import
from builtins import object
from mock import Mock

from .fixtures.mock_cmdl import cmdl
from .fixtures.mock_utils import MockUtils

from relaax.server.rlx_server.rlx_config import options
from relaax.server.rlx_server.rlx_agent_proxy import RLXAgentProxy


class TestRLXAgentProxy(object):

    @classmethod
    def setup_class(cls):
        options.Agent = Mock()

    @classmethod
    def teardown_class(cls):
        cmdl.restore()

    def setup_method(self, method):
        self.ap = RLXAgentProxy()
        self.ap.agent = Mock()

    def test_agent_init_done(self):
        self.ap.agent.init = lambda exploit: True
        assert self.ap.init({'command': 'init', 'exploit': False}) == {'response': 'ready'}

    def test_agent_init_fail(self):
        self.ap.agent.init = lambda exploit: False
        assert self.ap.init({'command': 'init', 'exploit': False}) == {'response': 'error', 'message': 'can\'t initialize agent'}

    def test_agent_reset_done(self):
        self.ap.agent.reset = lambda: True
        assert self.ap.reset() == {'response': 'done'}

    def test_agent_reset_fail(self):
        self.ap.agent.reset = lambda: False
        assert self.ap.reset() == {'response': 'error', 'message': 'can\'t reset agent'}

    def test_agent_update_done(self):
        self.ap.agent.update = lambda reward, state, terminal: [10, 10]
        assert (self.ap.update({'reward': None, 'state': None, 'terminal': None})
                == {'response': 'action', 'data': [10, 10]})

    def test_agent_command_exception(self):
        self.ap.agent.init = lambda exploit: MockUtils.raise_(Exception('some init error'))
        res = self.ap.data_received({'command': 'init'})
        assert res == {'message': 'some init error', 'response': 'error'}

    def test_unknown_command_error(self):
        res = self.ap.data_received({'command': 'wrong_command'})
        assert res == {'response': 'error', 'message': 'unknown command'}
