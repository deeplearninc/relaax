from __future__ import absolute_import
from builtins import object
from . import ddpg_config
from .lib import ddpg_episode

from relaax.common import profiling


profiler = profiling.get_profiler(__name__)


# DA3CAgent implements training regime for DA3C algorithm
# If exploit on init set to True, agent will run in exploitation regime:
# stop updating shared parameters and at the end of every episode load
# new policy parameters from PS
class Agent(object):
    def __init__(self, parameter_server, metrics):
        self.ps = parameter_server
        self.metrics = metrics

    # environment is ready and
    # waiting for agent to initialize
    def init(self, exploit=False, hogwild_update=True):
        self.episode = ddpg_episode.DDPGEpisode(self.ps, self.metrics, exploit, hogwild_update)
        self.episode.begin()
        return True

    # environment generated new state and reward
    # and asking agent for an action for this state
    @profiler.wrap
    def update(self, reward, state, terminal):
        self.episode.step(reward, state, terminal)

        if terminal:
            self.episode.begin()

        return self.episode.last_action

    @staticmethod
    def check_state_shape(state):
        if state is None:
            return
        expected_shape = list(ddpg_config.options.algorithm.input.shape)
        actual_shape = list(np.asarray(state).shape)
        if actual_shape != expected_shape:
            logger.warning('State shape %s does not match to expected one %s.',
                           repr(actual_shape), repr(expected_shape))
