from __future__ import absolute_import
from relaax.server.parameter_server import parameter_server_base
from relaax.server.common import session

from . import dqn_model


class ParameterServer(parameter_server_base.ParameterServerBase):
    def init_session(self):
        self.session = session.Session(dqn_model.GlobalServer())
        self.session.op_initialize()

    def n_step(self):
        return self.session.op_n_step()

    def close(self):
        self.session.close()

    def create_checkpoint(self):
        return self.session.create_checkpoint()

    def get_session(self):
        return self.session
