from __future__ import absolute_import

from builtins import object
import logging
import numpy as np
import scipy.signal
import threading
import six.moves.queue as queue

from relaax.common import profiling
from relaax.server.common import session
from relaax.common.algorithms.lib import utils
from relaax.common.algorithms.lib import observation

from .lib.da3c_replay_buffer import DA3CReplayBuffer
from . import da3c_config
from . import da3c_model


logger = logging.getLogger(__name__)
profiler = profiling.get_profiler(__name__)
M = False


# DA3CAgent implements training regime for DA3C algorithm
# If exploit on init set to True, agent will run in exploitation regime:
# stop updating shared parameters and at the end of every episode load
# new policy parameters from PS
class Agent(object):
    def __init__(self, parameter_server, metrics):
        self.ps = parameter_server
        self.metrics = metrics
        self.exploit = False
        self.session = None
        self.lstm_zero_state = None
        self.lstm_state = self.initial_lstm_state = None
        self.observation = None
        self.last_action = None
        self.last_value = None
        self.last_probs = None
        self.queue = None
        self.icm_observation = None
        self.replay_buffer = None
        self.terminal = False
        self.discounted_reward = None

    # environment is ready and
    # waiting for agent to initialize
    def init(self, exploit=False):
        self.exploit = exploit
        model = da3c_model.AgentModel()
        self.session = session.Session(model)
        if da3c_config.config.use_lstm:
            self.lstm_state = self.initial_lstm_state = self.lstm_zero_state = model.lstm_zero_state

        self.observation = observation.Observation(da3c_config.config.input.history)
        self.last_action = None
        self.last_value = None
        self.last_probs = None
        if da3c_config.config.hogwild and not da3c_config.config.use_icm:
            self.queue = queue.Queue(10)
            threading.Thread(target=self.execute_tasks).start()
            self.receive_experience()
        else:
            self.queue = None
        if da3c_config.config.use_icm:
            self.icm_observation = observation.Observation(da3c_config.config.input.history)

        self.replay_buffer = DA3CReplayBuffer(self)
        return True

    # Callback methods
    def begin(self):
        self.do_task(self.receive_experience)
        if da3c_config.config.use_lstm:
            self.initial_lstm_state = self.lstm_state
        self.get_action_and_value()

    def end(self, experience):
        if not self.exploit:
            self.do_task(lambda: self.send_experience(experience))

    @profiler.wrap
    def reset(self):
        if da3c_config.config.use_lstm:
            self.initial_lstm_state = self.lstm_state = self.lstm_zero_state
    # End callback methods

    @profiler.wrap
    def step(self, reward, state, terminal):
        if reward is not None:
            reward = np.tanh(reward)
            if da3c_config.config.use_icm:
                reward += self.get_intrinsic_reward(state)
            self.push_experience(reward, terminal)
        else:
            if da3c_config.config.use_icm:
                self.icm_observation.add_state(None)
                self.icm_observation.add_state(state)

        if terminal:
            self.observation.add_state(None)
        else:
            assert state is not None
            self.metrics.histogram('state', state)
            self.observation.add_state(state)

        self.terminal = terminal
        assert self.last_action is None
        assert self.last_value is None
        assert self.last_probs is None

        self.get_action_and_value()

    @property
    def experience(self):
        return self.replay_buffer.experience

    # environment generated new state and reward
    # and asking agent for an action for this state
    @profiler.wrap
    def update(self, reward, state, terminal):
        self.check_state_shape(state)
        self.step(reward, state, terminal)
        return self.last_action

    @staticmethod
    def check_state_shape(state):
        if state is None:
            return
        expected_shape = list(da3c_config.options.algorithm.input.shape)
        actual_shape = list(np.asarray(state).shape)
        if actual_shape != expected_shape:
            logger.warning('State shape %s does not match to expected one %s.', repr(actual_shape),
                           repr(expected_shape))

    #########################
    # From batch

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
        self.apply_gradients(self.compute_gradients(experience), len(experience))
        if da3c_config.config.use_icm:
            self.ps.session.op_icm_apply_gradients(gradients=self.compute_icm_gradients(experience))

    @profiler.wrap
    def receive_experience(self):
        self.ps.session.op_check_weights()
        weights = self.ps.session.op_get_weights()
        if M:
            for i, w in enumerate(utils.Utils.flatten(weights)):
                self.metrics.histogram('weight_%d' % i, w)
        self.session.op_assign_weights(weights=weights)
        if da3c_config.config.use_icm:
            self.session.op_icm_assign_weights(weights=self.ps.session.op_icm_get_weights())

    def push_experience(self, reward, terminal):
        assert self.observation.queue is not None
        assert self.last_action is not None
        assert self.last_value is not None
        assert self.last_probs is not None

        self.replay_buffer.step(
            terminal,
            reward=reward,
            state=self.observation.queue,
            action=self.last_action,
            value=self.last_value,
            probs=self.last_probs
        )
        self.last_action = None
        self.last_value = None
        self.last_probs = None

    def get_action_and_value(self):
        if self.observation.queue is None:
            self.last_action = None
            self.last_value = None
            self.last_probs = None
        else:
            self.last_action, self.last_value = self.get_action_and_value_from_network()
            assert self.last_action is not None
            assert self.last_value is not None
            assert self.last_probs is not None

    def get_action_and_value_from_network(self):
        if da3c_config.config.use_lstm:
            action, value, lstm_state = \
                    self.session.op_get_action_value_and_lstm_state(state=[self.observation.queue],
                                                                    lstm_state=self.lstm_state,
                                                                    lstm_step=[1])
            condition = self.experience is not None and (len(self.experience) ==
                                                         da3c_config.config.batch_size or self.terminal)
            if not condition:
                self.lstm_state = lstm_state
        else:
            action, value = self.session.op_get_action_and_value(state=[self.observation.queue])

        value, = value
        if len(action) == 1:
            if M:
                self.metrics.histogram('action', action)
            self.last_probs, = action
            return utils.choose_action_descrete(self.last_probs), value
        mu, sigma2 = action
        self.last_probs = mu
        if M:
            self.metrics.histogram('mu', mu)
            self.metrics.histogram('sigma2', sigma2)
        return utils.choose_action_continuous(mu, sigma2, da3c_config.config.output.action_low,
                                              da3c_config.config.output.action_high), value

    def get_intrinsic_reward(self, state):
        self.icm_observation.add_state(state)

        if state is not None:
            icm_input = [self.observation.queue, self.icm_observation.queue]
            return self.session.op_get_intrinsic_reward(state=icm_input, probs=[self.last_probs])[0]
        return 0

    def compute_gradients(self, experience):
        r = 0.0
        if self.last_value is not None:
            r = self.last_value

        reward = experience['reward']
        gamma = da3c_config.config.rewards_gamma

        # compute discounted rewards
        self.discounted_reward = self.discount(np.asarray(reward + [r], dtype=np.float32), gamma)[:-1]

        if da3c_config.config.use_gae:
            forward_values = np.asarray(experience['value'][1:] + [r]) * gamma
            rewards = np.asarray(reward) + forward_values - np.asarray(experience['value'])

            gae_gamma = da3c_config.config.rewards_gamma * da3c_config.config.gae_lambda
            advantage = self.discount(rewards, gae_gamma)

        feeds = dict(state=experience['state'], action=experience['action'],
                     value=experience['value'], discounted_reward=self.discounted_reward)

        if da3c_config.config.use_lstm:
            feeds.update(dict(lstm_state=self.initial_lstm_state, lstm_step=[len(reward)]))
        if da3c_config.config.use_gae:
            feeds.update(dict(advantage=advantage))

        gradients, summaries = self.session.op_compute_gradients_and_summaries(**feeds)
        self.metrics.summary(summaries)
        return gradients

    def compute_icm_gradients(self, experience):
        states, icm_states = experience['state'], []
        for i in range(len(states) - 1):
            icm_states.extend((states[i], states[i + 1]))
        icm_states.extend((states[-1], self.icm_observation.queue))
        return self.session.op_compute_icm_gradients(state=icm_states,
                                                     action=experience['action'],
                                                     probs=experience['probs'])

    def apply_gradients(self, gradients, experience_size):
        if M:
            for i, g in enumerate(utils.Utils.flatten(gradients)):
                self.metrics.histogram('gradients_%d' % i, g)
        self.ps.session.op_apply_gradients(gradients=gradients, increment=experience_size)
        self.ps.session.op_check_weights()

    @staticmethod
    def discount(x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
