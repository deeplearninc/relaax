from __future__ import absolute_import

import logging
import numpy as np

from relaax.common.algorithms.lib import utils
from . import pg_config

logger = logging.getLogger(__name__)


# PGAgent implements training regime for Policy Gradient algorithm
# If exploit on init set to True, agent will run in exploitation regime:
# stop updating shared parameters and at the end of every episode load
# new policy parameters from PS
class ModelApi(object):
    def __init__(self, session_arg, exploit, parameter_server):
        self.ps = parameter_server
        self.exploit = exploit
        self.session = session_arg

    def load_shared_parameters(self):
        self.session.op_assign_weights(weights=self.ps.session.op_get_weights())

    def action_from_policy(self, state):
        assert state is not None
        state = np.asarray(state)
        state = np.reshape(state, (1, ) + state.shape)
        probabilities, = self.session.op_get_action(state=state)
        return utils.choose_action_descrete(probabilities, self.exploit)

    def compute_gradients(self, experience_arg):
        discounted_reward = utils.discounted_reward(experience_arg['reward'], pg_config.config.GAMMA)
        return self.session.op_compute_gradients(state=experience_arg['state'],
                                                 action=experience_arg['action'],
                                                 discounted_reward=discounted_reward)

    def apply_gradients(self, gradients, size):
        self.ps.session.op_apply_gradients(gradients=gradients, increment=size)
