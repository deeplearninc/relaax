from __future__ import absolute_import
from builtins import object
from . import trpo_config

from .old_trpo_gae.agent import agent


class Agent(object):
    def __init__(self, parameter_server):
        self.ps = parameter_server
        self.metrics = parameter_server.metrics
        self.agent = agent.Agent(trpo_config.options, parameter_server)

    # environment is ready and
    # waiting for agent to initialize
    def init(self, exploit=False):
        return True

    # environment generated new state and reward
    # and asking agent for an action for this state
    def update(self, reward, state, terminal):
        return None

    # environment is asking to reset agent
    def reset(self):
        return True
