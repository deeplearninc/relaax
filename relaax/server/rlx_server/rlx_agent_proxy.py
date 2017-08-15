from __future__ import absolute_import
from builtins import str
from builtins import object
import logging
import traceback
import numpy as np

from relaax.server.common.bridge import ps_bridge_connection
from relaax.server.common.bridge import metrics_bridge_connection
from relaax.server.common.metrics import x_metrics
from .rlx_config import options

log = logging.getLogger(__name__)


class RLXAgentProxy(object):

    def __init__(self):
        ps = ps_bridge_connection.PsBridgeConnection(options.parameter_server)
        metrics_connection = metrics_bridge_connection.MetricsBridgeConnection(options)
        metrics = x_metrics.XMetrics(ps.session.op_n_step, metrics_connection.metrics)
        self.agent = options.Agent(ps, metrics)

    def init(self, data):
        exploit = data['exploit'] if 'exploit' in data else False
        if self.agent.init(exploit):
            return {'response': 'ready'}
        else:
            return self._error_message('can\'t initialize agent')

    def update(self, data):
        action = self.agent.update(
            reward=data['reward'],
            state=data['state'],
            terminal=data['terminal']
        )
        if isinstance(action, np.ndarray):
            action = np.asarray(action).tolist()
        return {'response': 'action', 'data': action}

    def reset(self, ignore=None):
        if self.agent.reset():
            return {'response': 'done'}
        else:
            return self._error_message('can\'t reset agent')

    def update_metrics(self, data):
        self.agent.metrics.update(data['data'])
        return {'response': 'done'}

    def data_received(self, data):
        try:
            if data['command'] in ['init', 'update', 'reset', 'update_metrics']:
                return getattr(self, data['command'])(data)
            else:
                return self._error_message('unknown command')
        except BaseException as e:
            log.error("Error while processing [%s] command by the agent" % data.get('command'))
            log.debug(traceback.format_exc())
            return self._error_message(str(e))

    def _error_message(self, message):
        return {'response': 'error', 'message': message}
