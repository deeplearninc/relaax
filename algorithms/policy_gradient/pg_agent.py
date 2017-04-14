import numpy as np

from relaax.server.common import session

from lib import episode
from lib import utils

import pg_config
import pg_model


# PGAgent implements training regime for Policy Gradient algorithm
# If exploit on init set to True, agent will run in exploitation regime:
# stop updating shared parameters and at the end of every episode load
# new policy parameters from PS
class PGAgent(object):

    def __init__(self, parameter_server):
        self.ps = parameter_server

    # environment is ready and
    # waiting for agent to initialize
    def init(self, exploit=False):
        # Initialize TF
        self.session = session.Session(pg_model.PolicyModel())
        self.episode = None if exploit else episode.Episode()
        if self.episode is None:
            self.load_shared_parameters()
        return True

    # environment generated new state and reward
    # and asking agent for an action for this state
    def update(self, reward, state, terminal):
        assert (state is None) == terminal

        if self.episode is not None and not self.episode.in_episode:
            self.load_shared_parameters()
            self.episode.begin()

        if state is None:
            action = None
        else:
            action = self.action_from_policy(state)
            assert action is not None

        if self.episode is not None:
            self.episode.step(reward, state, action)

        if self.episode is not None:
            if (self.episode.length == pg_config.config.batch_size) or terminal:
                experience = self.episode.end()
                self.apply_gradients(self.compute_gradients(experience))

        return action

    # environment is asking to reset agent
    def reset(self):
        if self.episode is not None:
            self.episode = episode.Episode()
        return True

# Helper methods

    # reload policy weights from PS
    def load_shared_parameters(self):
        self.session.op_assign_weights(values=self.ps.op_get_weights())

    def action_from_policy(self, state):
        assert state is not None

        probabilities, = self.session.op_get_action(state=[state])
        return utils.choose_action(probabilities)

    def compute_gradients(self, experience):
        discounted_reward = utils.discounted_reward(
            experience.rewards,
            pg_config.config.GAMMA
        )
        return self.session.op_compute_gradients(
            state=experience.states,
            action=experience.actions,
            discounted_reward=discounted_reward
        )

    def apply_gradients(self, gradients):
        self.ps.op_apply_gradients(gradients=gradients)
