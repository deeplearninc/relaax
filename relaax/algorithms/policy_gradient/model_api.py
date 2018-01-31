from __future__ import absolute_import

import logging

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
        self.agent_weights_id = 0

    def load_shared_parameters(self):
        weights, self.agent_weights_id = self.ps.session.op_get_weights_signed()
        self.session.op_assign_weights(weights=weights)

    def action_from_policy(self, state):
        probabilities, = self.session.op_get_action(state=[state])
        return utils.choose_action_descrete(probabilities, self.exploit)

    def compute_gradients(self, experience_arg):
        discounted_reward = utils.discounted_reward(experience_arg['reward'], pg_config.config.rewards_gamma)
        return self.session.op_compute_gradients(state=experience_arg['state'],
                                                 action=experience_arg['action'],
                                                 discounted_reward=discounted_reward)

    def apply_gradients(self, gradients, size):
        self.ps.session.op_submit_gradients(gradients=gradients, step_inc=size,
                                            agent_step=self.agent_weights_id)

    def compute_and_apply_gradients(self, experience):
        experience_size = len(experience)
        self.apply_gradients(self.compute_gradients(experience), experience_size)

        scaled_rewards_sum = sum(experience['reward']) * (experience_size / pg_config.config.batch_size)
        self.ps.session.op_add_rewards_to_model_score_routine(reward_sum=scaled_rewards_sum,
                                                              reward_weight=experience_size)
