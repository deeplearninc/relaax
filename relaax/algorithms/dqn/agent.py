from __future__ import absolute_import
from builtins import object
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
    def update(self, reward, state, terminal):
        self.episode.step(reward, state, terminal)

        return self.episode.last_action
