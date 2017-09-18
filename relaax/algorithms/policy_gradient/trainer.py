from __future__ import absolute_import

import logging
import numpy as np

from . import pg_config
from relaax.common.algorithms.lib import experience
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
            self.model_api.apply_gradients(self.model_api.compute_gradients(_experience), len(_experience))

    def reset(self):
        self.experience = None

    def step(self, reward, state, terminal):
        if reward is not None:
            self.push_experience(reward, terminal)
        if terminal and state is not None:
            logger.warning('PG Agent.step ignores state in case of terminal.')
            state = None
        else:
            assert state is not None
        if not terminal:
            state = np.asarray(state)
            if state.size == 0:
                state = np.asarray([0])
            state = np.reshape(state, state.shape + (1,))
        action = self.get_action(state)
        self.keep_state_and_action(state, action)
        return action

    # Helper methods
    def push_experience(self, reward, terminal):
        assert self.last_state is not None
        assert self.last_action is not None

        assert self.experience is not None
        self.experience.push_record(reward=reward, state=self.last_state, action=self.last_action)

        if (len(self.experience) == pg_config.config.batch_size) or terminal:
            self.end()
            if terminal:
                self.reset()
            self.begin()

        self.last_state = None
        self.last_action = None

    def get_action(self, state):
        if state is None:
            return None
        action = self.model_api.action_from_policy(state)
        assert action is not None
        return action

    def keep_state_and_action(self, state, action):
        assert self.last_state is None
        assert self.last_action is None

        self.last_state = state
        self.last_action = action
