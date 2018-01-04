from __future__ import absolute_import
from relaax.server.parameter_server import parameter_server_base
from relaax.server.common import session

from . import da3c_model


class ParameterServer(parameter_server_base.ParameterServerBase):
    def init_session(self):
        self.session = session.Session(da3c_model.SharedParameters())
        self.session.op_initialize()

    def n_step(self):
        return self.session.op_n_step()

    def score(self):
        return self.session.op_score()
