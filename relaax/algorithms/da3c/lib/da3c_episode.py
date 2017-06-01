from __future__ import absolute_import
from builtins import range
from builtins import object
import numpy as np

from relaax.server.common import session
from relaax.common.algorithms.lib import episode
from relaax.common.algorithms.lib import utils

from .. import da3c_config
from .. import da3c_model
from . import da3c_observation


class DA3CEpisode(object):
    def __init__(self, parameter_server, exploit):
        self.exploit = exploit
        self.ps = parameter_server
        model = da3c_model.AgentModel()
        self.session = session.Session(model)
        self.reset()
        self.observation = da3c_observation.DA3CObservation()
        self.last_action = None
        self.last_value = None
        if da3c_config.config.use_lstm:
            self.lstm_zero_state = model.lstm_zero_state
            self.lstm_state = model.lstm_zero_state

    @property
    def experience(self):
        return self.episode.experience

    def begin(self):
        self.load_shared_parameters()
        if da3c_config.config.use_lstm:
            self.lstm_state = self.lstm_zero_state
        self.get_action_and_value()
        self.episode.begin()

    def step(self, reward, state, terminal):
        if reward is not None:
            self.push_experience(reward)
        assert (state is None) == terminal
        self.observation.add_state(state)

        assert self.last_action is None
        assert self.last_value is None

        self.get_action_and_value()

    def end(self):
        experience = self.episode.end()
        if not self.exploit:
            self.apply_gradients(self.compute_gradients(experience), len(experience))

    def reset(self):
        self.episode = episode.Episode('reward', 'state', 'action', 'value')

    # Helper methods

    def push_experience(self, reward):
        assert self.observation.queue is not None
        assert self.last_action is not None
        assert self.last_value is not None

        self.episode.step(
            reward=reward,
            state=self.observation.queue,
            action=self.last_action,
            value=self.last_value
        )
        self.last_action = None
        self.last_value = None

    def get_action_and_value(self):
        if self.observation.queue is None:
            self.last_action = None
            self.last_value = None
        else:
            self.last_action, self.last_value = self.get_action_and_value_from_network()
            assert self.last_action is not None
            assert self.last_value is not None

    def keep_action_and_value(self, action, value):
        assert self.last_action is None
        assert self.last_value is None

        self.last_action = action
        self.last_value = value

    def load_shared_parameters(self):
        self.session.op_assign_weights(weights=self.ps.session.op_get_weights())

    def get_action_and_value_from_network(self):
        if da3c_config.config.use_lstm:
            action, value, self.lstm_state = self.session.op_get_action_value_and_lstm_state(
                    state=[self.observation.queue], lstm_state=self.lstm_state, lstm_step=[1])
        else:
            action, value = self.session.op_get_action_and_value(
                    state=[self.observation.queue])
        value, = value
        if len(action) == 1:
            probabilities, = action
            return utils.choose_action_descrete(probabilities), value
        mu, sigma2 = action
        return utils.choose_action_continuous(mu, sigma2), value

    def compute_gradients(self, experience):
        r = 0.0
        if self.last_value is not None:
            r = self.last_value

        reward = experience['reward']
        discounted_reward = np.zeros_like(reward, dtype=np.float32)

        # compute and accumulate gradients
        for t in reversed(range(len(reward))):
            r = reward[t] + da3c_config.config.rewards_gamma * r
            discounted_reward[t] = r

        if da3c_config.config.use_lstm:
            return self.session.op_compute_gradients(
                    state=experience['state'], action=experience['action'],
                    value=experience['value'], discounted_reward=discounted_reward,
                    lstm_state=self.lstm_state, lstm_step=[len(reward)])

        return self.session.op_compute_gradients(
                state=experience['state'], action=experience['action'],
                value=experience['value'], discounted_reward=discounted_reward)

    def apply_gradients(self, gradients, experience_size):
        self.ps.session.op_apply_gradients(
            gradients=gradients,
            increment=experience_size
        )
