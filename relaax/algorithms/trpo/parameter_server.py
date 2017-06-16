from __future__ import absolute_import

from relaax.server.parameter_server import parameter_server_base
from relaax.server.common import session

from . import trpo_config
from . import trpo_model
from .old_trpo_gae.parameter_server import parameter_server


class ParameterServer(parameter_server_base.ParameterServerBase):
    def close(self):
        self.session.close()

    def initialize_algorithm(self):
        sp = trpo_model.SharedParameters()
        self.session = session.Session(sp)
        sp._ps_bridge = parameter_server.ParameterServer(trpo_config.config, None, None, self.session).bridge()
        self.session.op_initialize()

    def make_checkpoint(self):
        return self.session.make_checkpoint()

    def get_session(self):
        return self.session

    def n_step(self):
        return self.session.op_n_step()
