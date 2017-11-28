from __future__ import absolute_import

from builtins import object
import numpy as np

from .lib import dqn_trainer
from relaax.common import profiling


profiler = profiling.get_profiler(__name__)


class Agent(object):
    def __init__(self, parameter_server, metrics):
        self.ps = parameter_server
        self.metrics = metrics

    # environment is ready and
    # waiting for agent to initialize
    def init(self, exploit=False):
        self.episode = dqn_trainer.Trainer(self.ps, self.metrics, exploit=exploit)
        self.episode.begin()
        return True

    # environment generated new state and reward
    # and asking agent for an action for this state
    @profiler.wrap
    def update(self, reward, state, terminal, info=None):
        # replace empty state with constant one
        if list(np.asarray(state).shape) == [0]:
            state = [0]

        skip = False
        if info is not None:
            if 'skip' in info:
                skip = info['skip']

        self.episode.step(reward, state, terminal, skip)

        return self.episode.last_action
