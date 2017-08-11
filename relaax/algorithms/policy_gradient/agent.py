from __future__ import absolute_import

from builtins import object
import logging
import numpy as np

from relaax.server.common import session
#from relaax.common.algorithms.lib import episode
from relaax.common.algorithms.lib import utils

from . import pg_config
from . import pg_model
#from .lib import pg_batch
from .lib.pg_replay_buffer import PGReplayBuffer


logger = logging.getLogger(__name__)


# PGAgent implements training regime for Policy Gradient algorithm
# If exploit on init set to True, agent will run in exploitation regime:
# stop updating shared parameters and at the end of every episode load
# new policy parameters from PS
class Agent(object):
    def __init__(self, parameter_server, metrics):
        self.ps = parameter_server
        self.metrics = metrics
        #self.episode = None
        self.exploit = False
        self.session = None
        self.last_state = None
        self.last_action = None
        self.replay_buffer = None

    # environment is ready and
    # waiting for agent to initialize
    def init(self, exploit=False):
        #self.batch = pg_batch.PGBatch(self.ps, exploit)
        self.exploit = exploit
        self.session = session.Session(pg_model.PolicyModel())
        #self.reset()

        #self.batch.begin()
        #self.begin()
        self.replay_buffer = PGReplayBuffer(self)

        return True

    # Callback methods
    def begin(self):
        self.load_shared_parameters()

    def end(self, experience):
        if not self.exploit:
            self.apply_gradients(self.compute_gradients(experience), len(experience))

    # environment is asking to reset agent
    def reset(self):
        pass

    # End callback methods

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

    @property
    def experience(self):
        return self.replay_buffer.experience

    # environment generated new state and reward
    # and asking agent for an action for this state
    def update(self, reward, state, terminal):
        self.check_state_shape(state)

        action = self.step(reward, state, terminal)

        # if (len(self.experience) == pg_config.config.batch_size) or terminal:
        #     self.end()
        #     self.begin()

        return action


    # Helper methods

    def push_experience(self, reward, terminal):
        assert self.last_state is not None
        assert self.last_action is not None

        self.replay_buffer.step(
            terminal,
            reward=reward,
            state=self.last_state,
            action=self.last_action
        )
        self.last_state = None
        self.last_action = None

    def get_action(self, state):
        if state is None:
            return None
        action = self.action_from_policy(state)
        assert action is not None
        return action

    def keep_state_and_action(self, state, action):
        assert self.last_state is None
        assert self.last_action is None

        self.last_state = state
        self.last_action = action

    def load_shared_parameters(self):
        self.session.op_assign_weights(weights=self.ps.session.op_get_weights())

    def action_from_policy(self, state):
        assert state is not None
        state = np.asarray(state)
        state = np.reshape(state, (1, ) + state.shape)
        probabilities, = self.session.op_get_action(state=state)
        return utils.choose_action_descrete(probabilities, self.exploit)

    def compute_gradients(self, experience):
        discounted_reward = utils.discounted_reward(
            experience['reward'],
            pg_config.config.GAMMA
        )
        return self.session.op_compute_gradients(
            state=experience['state'],
            action=experience['action'],
            discounted_reward=discounted_reward
        )

    def apply_gradients(self, gradients, size):
        self.ps.session.op_apply_gradients(gradients=gradients, increment=size)

    @staticmethod
    def check_state_shape(state):
        if state is None:
            return
        expected_shape = list(pg_config.options.algorithm.input.shape)
        actual_shape = list(np.asarray(state).shape)
        if actual_shape != expected_shape:
            logger.warning('State shape %s does not match to expected one %s.',
                           repr(actual_shape), repr(expected_shape))
