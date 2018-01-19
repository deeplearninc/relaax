from __future__ import absolute_import

import logging

from . import pg_config
from relaax.common.algorithms.lib import experience
from relaax.common.algorithms.lib import observation
from .model_api import ModelApi

logger = logging.getLogger(__name__)


# PG Trainer implements training regime for Policy Gradient algorithm
# If exploit on init set to True, agent will run in exploitation regime:
# stop updating shared parameters and at the end of every episode load
# new policy parameters from PS
class Trainer(object):
    def __init__(self, session_arg, exploit, parameter_server, metrics):
        self.ps = parameter_server
        self.metrics = metrics
        self.exploit = exploit
        self.last_state = None
        self.last_action = None
        self.keys = ['reward', 'state', 'action']
        self.experience = None
        self.model_api = ModelApi(session_arg, exploit, parameter_server)
        self.observation = observation.Observation(pg_config.config.input.history)
        self.begin()

    def begin(self):
        assert self.experience is None
        self.model_api.load_shared_parameters()
        self.experience = experience.Experience(*self.keys)

    def end(self):
        assert self.experience is not None
        _experience = self.experience
        self.experience = None

        if not self.exploit:
            self.model_api.compute_and_apply_gradients(_experience)

    def reset(self):
        self.experience = None

    def step(self, reward, state, terminal):
        if reward is not None:
            self.push_experience(reward, terminal)

        if terminal:
            self.observation.add_state(None)
        else:
            assert state is not None
            self.metrics.histogram('state', state)
            self.observation.add_state(state)

        action = self.get_action()
        self.keep_action(action)
        return action

    # Helper methods
    def push_experience(self, reward, terminal):
        assert self.observation.queue is not None
        assert self.last_action is not None

        assert self.experience is not None
        self.experience.push_record(reward=reward, state=self.observation.queue, action=self.last_action)

        if (len(self.experience) == pg_config.config.batch_size) or terminal:
            self.end()
            if terminal:
                self.reset()
            self.begin()

        self.last_action = None

    def get_action(self):
        if self.observation.queue is None:
            return None
        action = self.model_api.action_from_policy(self.observation.queue)
        assert action is not None
        return action

    def keep_action(self, action):
        assert self.last_action is None
        self.last_action = action
