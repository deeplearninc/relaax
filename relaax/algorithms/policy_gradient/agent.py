from __future__ import absolute_import

from builtins import object
import logging
import numpy as np

from . import pg_config
from .lib import pg_episode


logger = logging.getLogger(__name__)


# PGAgent implements training regime for Policy Gradient algorithm
# If exploit on init set to True, agent will run in exploitation regime:
# stop updating shared parameters and at the end of every episode load
# new policy parameters from PS
class Agent(object):
    def __init__(self, parameter_server, metrics):
        self.ps = parameter_server
        self.metrics = metrics

    # environment is ready and
    # waiting for agent to initialize
    def init(self, exploit=False):
        self.episode = pg_episode.PGEpisode(self.ps, exploit)
        self.episode.begin()
        return True

    # environment generated new state and reward
    # and asking agent for an action for this state
    def update(self, reward, state, terminal):
        self.check_state_shape(state)

        action = self.episode.step(reward, state, terminal)

        if (len(self.episode.experience) == pg_config.config.batch_size) or terminal:
            self.episode.end()
            self.episode.begin()

        return action

    # environment is asking to reset agent
    def reset(self):
        self.episode.reset()
        return True

    @staticmethod
    def check_state_shape(state):
        if state is None:
            return
        expected_shape = list(pg_config.options.algorithm.input.shape)
        actual_shape = list(np.asarray(state).shape)
        if actual_shape != expected_shape:
            logger.warning('State shape %s does not match to expected one %s.',
                           repr(actual_shape), repr(expected_shape))
