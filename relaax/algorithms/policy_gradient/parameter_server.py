from __future__ import absolute_import
from relaax.server.parameter_server import parameter_server_base
from relaax.server.common import session

from . import pg_model


class ParameterServer(parameter_server_base.ParameterServerBase):
    def __init__(self, saver_factory, metrics_factory):
        self.session = session.Session(pg_model.SharedParameters())
        self.session.op_initialize()
        super(ParameterServer, self).__init__(saver_factory, metrics_factory)

    def close(self):
        self.session.close()

    def make_checkpoint(self):
        return self.session.make_checkpoint()

    def get_session(self):
        return self.session

    def n_step(self):
        return self.session.op_n_step()
