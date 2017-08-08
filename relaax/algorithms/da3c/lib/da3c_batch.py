from __future__ import absolute_import

from builtins import range
from builtins import object

import logging
import numpy as np
import scipy.signal
import six.moves.queue as queue
import threading

from relaax.common import profiling
from relaax.server.common import session
from relaax.common.algorithms.lib import episode
from relaax.common.algorithms.lib import utils

from .. import da3c_config
from .. import da3c_model
from . import da3c_observation


M = False

logger = logging.getLogger(__name__)
profiler = profiling.get_profiler(__name__)


class DA3CBatch(object):
    def __init__(self, parameter_server, metrics, exploit, hogwild_update):
        self.exploit = exploit
        self.ps = parameter_server
        self.metrics = metrics
        model = da3c_model.AgentModel()
        self.session = session.Session(model)
        if da3c_config.config.use_lstm:
            self.lstm_zero_state = model.lstm_zero_state
            self.lstm_state = self.initial_lstm_state = model.lstm_zero_state
        self.reset()
        self.observation = da3c_observation.DA3CObservation()
        self.last_action = None
        self.last_value = None
        if hogwild_update:
            self.queue = queue.Queue(10)
            threading.Thread(target=self.execute_tasks).start()
            self.receive_experience()
        else:
            self.queue = None
        if da3c_config.config.use_icm:
            self.icm_observation = da3c_observation.DA3CObservation()

    @property
    def experience(self):
        return self.episode.experience

    @profiler.wrap
    def begin(self):
        self.do_task(self.receive_experience)
        if da3c_config.config.use_lstm:
            self.initial_lstm_state = self.lstm_state
        self.get_action_and_value()
        self.episode.begin()

    @profiler.wrap
    def step(self, reward, state, terminal):
        if reward is not None:
            reward = reward / 100
            if da3c_config.config.use_icm:
                reward += self.get_intrinsic_reward(state)
            self.push_experience(reward)
        if terminal:
            if state is not None:
                logger.warning('DA3CBatch.step ignores state in case of terminal.')
                state = None
        else:
            assert state is not None
        if M:
            if state is not None:
                self.metrics.histogram('state', state)
        self.observation.add_state(state)

        self.terminal = terminal
        assert self.last_action is None
        assert self.last_value is None

        self.get_action_and_value()

    @profiler.wrap
    def end(self):
        experience = self.episode.end()
        if not self.exploit:
            self.do_task(lambda: self.send_experience(experience))

    @profiler.wrap
    def reset(self):
        self.episode = episode.Episode('reward', 'state', 'action', 'value')
        if da3c_config.config.use_lstm:
            self.initial_lstm_state = self.lstm_state = self.lstm_zero_state

    # Helper methods

    def execute_tasks(self):
        while True:
            task = self.queue.get()
            task()

    def do_task(self, f):
        if self.queue is None:
            f()
        else:
            self.queue.put(f)

    @profiler.wrap
    def send_experience(self, experience):
        actor_gradients, critic_gradients = self.compute_gradients(experience)
        self.apply_gradients(actor_gradients, critic_gradients, len(experience))
        if da3c_config.config.use_icm:
            self.ps.session.op_icm_apply_gradients(
                gradients=self.compute_icm_gradients(experience))

    @profiler.wrap
    def receive_experience(self):
        if False:
            self.ps.session.op_check_weights()
        actor_weights, critic_weights = self.ps.session.op_get_weights()
        if M:
            for i, w in enumerate(utils.Utils.flatten((actor_weights, critic_weights))):
                self.metrics.histogram('weight_%d' % i, w)
        self.session.op_assign_weights(actor_weights=actor_weights, critic_weights=critic_weights)
        if da3c_config.config.use_icm:
            self.session.op_icm_assign_weights(weights=self.ps.session.op_icm_get_weights())

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

    def get_action_and_value_from_network(self):
        if da3c_config.config.use_lstm:
            action, value, lstm_state = \
                self.session.op_get_action_value_and_lstm_state(state=[self.observation.queue],
                                                                lstm_state=self.lstm_state)
            condition = self.episode.experience is not None and \
                        (len(self.episode.experience) == da3c_config.config.batch_size or self.terminal)
            if not condition:
                self.lstm_state = lstm_state
        else:
            action, value = self.session.op_get_action_and_value(state=[self.observation.queue])

        value, = value
        if len(action) == 1:
            if M:
                self.metrics.histogram('action', action)
            probabilities, = action
            return utils.choose_action_descrete(probabilities), value
        mu, sigma2 = action
        if M:
            self.metrics.histogram('mu', mu)
            self.metrics.histogram('sigma2', sigma2)
        action = utils.choose_action_continuous(mu, sigma2,
                                              da3c_config.config.output.action_low,
                                              da3c_config.config.output.action_high)
        if M:
            self.metrics.histogram('action', action)
            self.metrics.histogram('value', value)
        return action, value

    def get_intrinsic_reward(self, state):
        self.icm_observation.add_state(state)

        if state is not None:
            icm_input = [self.observation.queue, self.icm_observation.queue]
            intrinsic_reward = self.session.op_get_intrinsic_reward(state=icm_input)

            print('intrinsic_reward', intrinsic_reward.shape, intrinsic_reward)
            return intrinsic_reward
        return 0

    def compute_gradients(self, experience):
        r = 0.0
        if self.last_value is not None:
            r = self.last_value

        reward = experience['reward']
        self.discounted_reward = np.zeros_like(reward, dtype=np.float32)

        gamma = da3c_config.config.rewards_gamma
        # compute discounted rewards
        for t in reversed(range(len(reward))):
            r = reward[t] + gamma * r
            self.discounted_reward[t] = r

        if da3c_config.config.use_gae:
            forward_values = np.asarray(experience['value'][1:] + [r]) * gamma
            rewards = np.asarray(reward) + forward_values - np.asarray(experience['value'])
            gae_gamma = da3c_config.config.rewards_gamma * da3c_config.config.gae_lambda
            advantage = self.discount(rewards, gae_gamma)

        feeds = dict(state=experience['state'], action=experience['action'],
                     discounted_reward=self.discounted_reward)

        if da3c_config.config.use_lstm:
            feeds.update(dict(lstm_state=self.initial_lstm_state))
        if da3c_config.config.use_gae:
            feeds.update(dict(advantage=advantage))

        actor_gradients, critic_gradients, summaries = \
                self.session.op_compute_gradients_and_summaries(**feeds)
        self.metrics.summary(summaries)
        return actor_gradients, critic_gradients

    def compute_icm_gradients(self, experience):
        states, icm_states = experience['state'], []
        for i in range(len(states) - 1):
            icm_states.extend((states[i], states[i + 1]))
        return self.session.op_compute_icm_gradients(
            state=icm_states, action=experience['action'],
            discounted_reward=self.discounted_reward)

    def apply_gradients(self, actor_gradients, critic_gradients, experience_size):
        if M:
            for i, g in enumerate(utils.Utils.flatten((actor_gradients, critic_gradients))):
                self.metrics.histogram('gradients_%d' % i, g)
        self.ps.session.op_apply_gradients(actor_gradients=actor_gradients,
                                           critic_gradients=critic_gradients, increment=experience_size)
        if False:
            self.ps.session.op_check_weights()

    @staticmethod
    def discount(x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
