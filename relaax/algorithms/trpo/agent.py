from __future__ import absolute_import

from builtins import object
import logging
import numpy as np

from relaax.server.common import session
from relaax.common import profiling

from . import trpo_config
from . import trpo_model
from .old_trpo_gae.agent import agent


logger = logging.getLogger(__name__)
profiler = profiling.get_profiler(__name__)


class Agent(object):
    def __init__(self, parameter_server, metrics):
        self.ps = parameter_server
        self.metrics = metrics

    # environment is ready and
    # waiting for agent to initialize
    def init(self, exploit=False):
        model = trpo_model.AgentModel()
        self.session = session.Session(model)
        self.agent = agent.Agent(self.ps, self.session)
        return True

    # environment generated new state and reward
    # and asking agent for an action for this state
    @profiler.wrap
    def update(self, reward, state, terminal):
        self.check_state_shape(state)
        if reward is not None:
            if self.agent.reward(reward):
                return None
        if terminal and state is not None:
            logger.warning('Agent.update ignores state in case of terminal.')
        else:
            assert (state is None) == terminal
        if terminal:
            self.agent.reset()
            return None
        return self.agent.act(np.asarray(state))

    # environment is asking to reset agent
    def reset(self):
        self.agent.reset()
        return True

    @staticmethod
    def check_state_shape(state):
        if state is None:
            return
        expected_shape = list(trpo_config.options.algorithm.input.shape)
        actual_shape = list(np.asarray(state).shape)
        if actual_shape != expected_shape:
            logger.warning('State shape %s does not match to expected one %s.',
                    repr(actual_shape), repr(expected_shape))
