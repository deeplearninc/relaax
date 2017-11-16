from __future__ import absolute_import
from relaax.server.parameter_server import parameter_server_base
from relaax.server.common import session

from . import dppo_model


class ParameterServer(parameter_server_base.ParameterServerBase):
    def init_session(self):
        policy_shared = dppo_model.SharedWeights(dppo_model.Network().weights)
        value_func_shared = dppo_model.SharedWeights(dppo_model.ValueNet().weights)

        """self.policy_session = session.Session(policy_shared)
        self.policy_session.op_initialize()
        self.policy_session.op_init_weight_history()

        self.value_session = session.Session(value_func_shared)
        self.value_session.op_initialize()
        self.value_session.op_init_weight_history()"""

        self.session = session.Session(policy=policy_shared, value_func=value_func_shared)

        self.session.policy.op_initialize()
        self.session.policy.op_init_weight_history()

        self.session.value_func.op_initialize()
        self.session.value_func.op_init_weight_history()

    def n_step(self):
        policy_step = self.session.policy.op_n_step()
        return policy_step
