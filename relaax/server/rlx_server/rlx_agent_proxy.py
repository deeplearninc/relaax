import logging
import traceback

from relaax.server.common.bridge.bridge_connection import BridgeConnection
from rlx_config import options

log = logging.getLogger(__name__)


class RLXAgentProxy(object):

    def __init__(self):
        connection = BridgeConnection(options.parameter_server)
        self.agent = options.algorithm_module.Agent(connection)

    def init(self, ignore=None):
        if self.agent.init():
            return {'response': 'ready'}
        else:
            return self._error_message('can\'t initialize agent')

    def update(self, data):
        action = self.agent.update(
            reward=data['reward'],
            state=data['state'],
            terminal=data['terminal']
        )
        return {'response': 'action', 'data': action}

    def reset(self, ignore=None):
        if self.agent.reset():
            return {'response': 'done'}
        else:
            return self._error_message('can\'t reset agent')

    def data_received(self, data):
        try:
            if data['command'] in ['init', 'update', 'reset']:
                return getattr(self, data['command'])(data)
            else:
                return self._error_message('unknown command')
        except BaseException as e:
            log.error("Error while processing [%s] command by the agent" % data.get('command'))
            log.error(traceback.format_exc())
            return self._error_message(str(e))

    def _error_message(self, message):
        return {'response': 'error', 'message': message}
