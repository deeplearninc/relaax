from __future__ import absolute_import
from relaax.server.parameter_server import parameter_server_base
from relaax.server.common import session

from .fun_model import GlobalWorkerNetwork, GlobalManagerNetwork


class ParameterServer(parameter_server_base.ParameterServerBase):
    def close(self):
        self.session.close()

    def initialize_algorithm(self):
        self.session = session.Session(
            gm=GlobalManagerNetwork(),
            gw=GlobalWorkerNetwork()
        )
        self.session.op_initialize()

    def make_checkpoint(self):
        return self.session.make_checkpoint()

    def get_session(self):
        return self.session

    def n_step(self):
        return self.session.op_n_step()
