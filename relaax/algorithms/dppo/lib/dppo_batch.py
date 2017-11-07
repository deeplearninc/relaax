from builtins import object

import logging
import numpy as np

from relaax.server.common import session
from relaax.common.algorithms.lib import episode
from relaax.common.algorithms.lib import utils

from relaax.algorithms.trpo.lib.core import ProbType, Categorical, DiagGauss

from .. import dppo_config
from .. import dppo_model

logger = logging.getLogger(__name__)


class DPPOBatch(object):
    def __init__(self, parameter_server, exploit):
        self.exploit = exploit
        self.ps = parameter_server
        model = dppo_model.Model(assemble_model=True)
        self.session = session.Session(policy=model.policy, value_func=model.value_func)
        self.episode = None

        if dppo_config.config.use_lstm:
            self.initial_lstm_state = self.lstm_state = self.lstm_zero_state = model.lstm_zero_state
        self.reset()

        self.last_state = None
        self.last_action = None
        self.last_prob = None

        self.policy_step = None
        self.value_step = None

        if dppo_config.config.output.continuous:
            self.prob_type = DiagGauss(dppo_config.config.output.action_size)
        else:
            self.prob_type = Categorical(dppo_config.config.output.action_size)

    @property
    def experience(self):
        return self.episode.experience

    def begin(self):
        self.load_shared_policy_parameters()
        self.load_shared_value_func_parameters()
        if dppo_config.config.use_lstm:
            self.initial_lstm_state = self.lstm_state
        self.episode.begin()

    def step(self, reward, state, terminal):
        if reward is not None:
            self.push_experience(reward)
        if terminal and state is not None:
            logger.debug('PGBatch.step ignores state in case of terminal.')
            state = None
        else:
            assert state is not None
        if not terminal:
            state = np.asarray(state)
            if state.size == 0:
                state = np.asarray([0])
            state = np.reshape(state, state.shape + (1,))
        action, prob = self.get_action_and_prob(state)
        self.keep_state_action_prob(state, action, prob)
        return action

    def end(self):
        experience = self.episode.end()
        if not self.exploit:
            # Send PPO policy gradient M times and value function gradient B times
            # from https://arxiv.org/abs/1707.02286

            for i in range(dppo_config.config.policy_iterations):
                self.apply_policy_gradients(self.compute_policy_gradients(experience), len(experience))
                self.load_shared_policy_parameters()
            logger.debug('Policy update finished')

            for i in range(dppo_config.config.value_func_iterations):
                self.apply_value_func_gradients(self.compute_value_func_gradients(experience), len(experience))
                self.load_shared_value_func_parameters()
            logger.debug('Value function update finished')

    def reset(self):
        self.episode = episode.Episode('reward', 'state', 'action', 'old_prob')
        if dppo_config.config.use_lstm:
            self.initial_lstm_state = self.lstm_state = self.lstm_zero_state

    # Helper methods

    def push_experience(self, reward):
        assert self.last_state is not None
        assert self.last_action is not None
        assert self.last_prob is not None

        self.episode.step(
            reward=reward,
            state=self.last_state,
            action=self.last_action,
            old_prob=self.last_prob
        )
        self.last_state = None
        self.last_action = None
        self.last_prob = None

    def get_action_and_prob(self, state):
        if state is None:
            return None, None
        action, prob = self.action_and_prob_from_policy(state)
        assert action is not None
        return action, prob

    def keep_state_action_prob(self, state, action, prob):
        assert self.last_state is None
        assert self.last_action is None
        assert self.last_prob is None

        self.last_state = state
        self.last_action = action
        self.last_prob = prob

    def load_shared_policy_parameters(self):
        # Load policy parameters from server if they are fresh
        new_policy_weights, new_policy_step = self.ps.session.policy.op_get_weights_signed()
        msg = "Current policy weights: {}, received weights: {}".format(self.policy_step, new_policy_step)

        if (self.policy_step is None) or (new_policy_step > self.policy_step):
            logger.debug(msg + ", updating weights")
            self.session.policy.op_assign_weights(weights=new_policy_weights)
            self.policy_step = new_policy_step
        else:
            logger.debug(msg + ", keeping old weights")

    def load_shared_value_func_parameters(self):
        # Load value function parameters from server if they are fresh
        new_value_func_weights, new_value_func_step = self.ps.session.value_func.op_get_weights_signed()
        msg = "Current value func weights: {}, received weights: {}".format(self.value_step, new_value_func_step)

        if (self.value_step is None) or (new_value_func_step > self.value_step):
            logger.debug(msg + ", updating weights")
            self.session.value_func.op_assign_weights(weights=new_value_func_weights)
            self.value_step = new_value_func_step
        else:
            logger.debug(msg + ", keeping old weights")

    def action_and_prob_from_policy(self, state):
        assert state is not None
        state = np.asarray(state)
        state = np.reshape(state, (1,) + state.shape)

        if dppo_config.config.use_lstm:
            # 0 <- actor's lstm state & critic's lstm state -> 1
            probabilities, self.lstm_state[0] = self.session.policy.op_get_action(state=state,
                                                                                  lstm_state=self.lstm_state[0],
                                                                                  lstm_step=[1])
        else:
            probabilities = self.session.policy.op_get_action(state=state)

        # logger.debug("probs: {}".format(probabilities))
        return self.prob_type.sample(probabilities)[0], probabilities[0]

    def compute_policy_gradients(self, experience):
        values = self.compute_state_values(experience['state'])
        advantages = MniAdvantage(experience['reward'], values, 0, dppo_config.config.gamma)

        feeds = dict(state=experience['state'], action=experience['action'],
                     advantage=advantages, old_prob=experience['old_prob'])
        if dppo_config.config.use_lstm:
            # 0 <- actor's lstm state & critic's lstm state -> 1
            feeds.update(dict(lstm_state=self.initial_lstm_state[0], lstm_step=[len(experience['state'])]))
        return self.session.policy.op_compute_ppo_clip_gradients(**feeds)

    def apply_policy_gradients(self, gradients, size):
        self.ps.session.policy.op_submit_gradients(gradients=gradients, step=self.policy_step)

    def compute_state_values(self, states):
        if dppo_config.config.use_lstm:
            # 0 <- actor's lstm state & critic's lstm state -> 1
            values, self.lstm_state[1] = self.session.value_func.op_value(state=states,
                                                                          lstm_state=self.initial_lstm_state[1],
                                                                          lstm_step=[len(states)])
        else:
            values = self.session.value_func.op_value(state=states)
        return values

    def compute_value_func_gradients(self, experience):
        # Hack to compute observed values for value function
        rwd = experience['reward']
        obs_values = MniAdvantage(rwd, np.zeros(len(rwd)), 0, dppo_config.config.gamma)

        feeds = dict(state=experience['state'], ytarg_ny=obs_values)
        if dppo_config.config.use_lstm:
            # 0 <- actor's lstm state & critic's lstm state -> 1
            feeds.update(dict(lstm_state=self.initial_lstm_state[1], lstm_step=[len(rwd)]))
        return self.session.value_func.op_compute_gradients(**feeds)

    def apply_value_func_gradients(self, gradients, size):
        self.ps.session.value_func.op_submit_gradients(gradients=gradients, step=self.value_step)


# Compute advantage estimates using rewards and value function predictions
# Mni et al, 2016
# For formula, see https://arxiv.org/pdf/1707.06347.pdf
def MniAdvantage(rewards, values, final_value, gamma):
    adv = np.zeros(len(rewards))
    rwd = rewards + [final_value]

    # TODO: optimize
    for i in range(len(adv)):
        adv[i] = -values[i]
        for j in range(i, len(rwd)):
            adv[i] += rwd[j] * np.power(gamma, j)

    return adv


# Advantage estimation using sub-segments of [0..T] interval
# See Appendix of https://arxiv.org/abs/1707.02286
def kStepReward(rewards, gamma, k):
    new_rewards = np.zeros(len(rewards))

    # TODO: optimize
    for i in range(len(rewards) // k):
        for j in range(k):
            idx = i * k + j
            new_rewards[idx] = rewards[idx] * np.pow(gamma, j)
