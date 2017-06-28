from __future__ import absolute_import
from builtins import object
from . import da3c_config
from .lib import da3c_episode

from relaax.common import profiling


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
        self.episode.step(reward, state, terminal)

        if len(self.episode.experience) == da3c_config.config.batch_size or terminal:
            self.episode.end()
            self.episode.begin()

        return self.episode.last_action

    # environment is asking to reset agent
    @profiler.wrap
    def reset(self):
        self.episode.reset()
        return True
