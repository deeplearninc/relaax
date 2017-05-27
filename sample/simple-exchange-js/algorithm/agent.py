from __future__ import absolute_import
from builtins import object

from relaax.server.common import session

from . import model


class Agent(object):
    def __init__(self, parameter_server):
        self.ps = parameter_server
        self.metrics = parameter_server.metrics
        self.session = session.Session(model.Model())

    def init(self, exploit=False):
        print('agent init(exploit=%s)' % exploit)
        return True

    def update(self, reward, state, terminal):
        print('agent update: ', reward, state, terminal)
        return self.session.op_get_action(state=state)

    def reset(self):
        return True
