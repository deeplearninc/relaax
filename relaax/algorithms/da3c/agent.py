from __future__ import absolute_import

from builtins import object
import logging
import numpy as np

from relaax.common import profiling

from . import da3c_config
from .lib import da3c_episode



logger = logging.getLogger(__name__)
profiler = profiling.get_profiler(__name__)


# DA3CAgent implements training regime for DA3C algorithm
# If exploit on init set to True, agent will run in exploitation regime:
# stop updating shared parameters and at the end of every episode load
# new policy parameters from PS
class Agent(object):
    def __init__(self, parameter_server):
        self.ps = parameter_server
        self.metrics = parameter_server.metrics

    # environment is ready and
    # waiting for agent to initialize
    def init(self, exploit=False, hogwild_update=True):
        self.episode = da3c_episode.DA3CEpisode(self.ps, exploit, hogwild_update)
        self.episode.begin()
        return True

    # environment generated new state and reward
    # and asking agent for an action for this state
    @profiler.wrap
    def update(self, reward, state, terminal):
        self.check_state_shape(state)
        self.episode.step(reward, state, terminal)

        if len(self.episode.experience) == da3c_config.config.batch_size or terminal:
            self.episode.end()
            if terminal:
                self.episode.reset()
            self.episode.begin()

        return self.episode.last_action

    @staticmethod
    def check_state_shape(state):
        if state is None:
            return
        expected_shape = list(da3c_config.options.algorithm.input.shape)
        actual_shape = list(np.asarray(state).shape)
        if actual_shape != expected_shape:
            logger.warning('State shape %s does not match to expected one %s.',
                    repr(actual_shape), repr(expected_shape))
