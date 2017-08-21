from __future__ import absolute_import
from relaax.server.parameter_server import parameter_server_base
from relaax.server.common import session

from . import ddpg_model


class ParameterServer(parameter_server_base.ParameterServerBase):
    '''
    def __init__(self, saver_factory, metrics_factory):
        self.session = session.Session(ddpg_model.SharedParameters())
        self.session.op_initialize()
        super(ParameterServer, self).__init__(saver_factory, metrics_factory)
        self.session.op_init_actor_target_weights()
        self.session.op_init_critic_target_weights()
    '''
    def init_session(self):
        self.session = session.Session(ddpg_model.SharedParameters())
        self.session.op_initialize()
        self.session.op_init_target_weights()

    def close(self):
        self.session.close()

    def create_checkpoint(self):
        return self.session.create_checkpoint()

    def get_session(self):
        return self.session

    def n_step(self):
        return self.session.op_n_step()
