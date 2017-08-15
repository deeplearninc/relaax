from __future__ import absolute_import
from builtins import object
from .lib import dqn_episode

from relaax.common import profiling


profiler = profiling.get_profiler(__name__)


class Agent(object):
    def __init__(self, parameter_server, metrics):
        self.ps = parameter_server
        self.metrics = metrics

    # environment is ready and
    # waiting for agent to initialize
    def init(self, exploit=False, hogwild_update=False):
        self.episode = dqn_episode.DQNEpisode(self.ps, self.metrics, exploit, hogwild_update)
        self.episode.begin()
        return True

    # environment generated new state and reward
    # and asking agent for an action for this state
    @profiler.wrap
    def update(self, reward, state, terminal):
        self.episode.step(reward, state, terminal)

        return self.episode.last_action
