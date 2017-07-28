from builtins import object

import logging
import numpy as np

from relaax.server.common import session
from relaax.common.algorithms.lib import episode
from relaax.common.algorithms.lib import utils
from relaax.common.algorithms.lib.batch import Batch

from .. import pg_config
from .. import pg_model


logger = logging.getLogger(__name__)


class PGBatch(Batch):
    def __init__(self, parameter_server, exploit):
        Batch.__init__(self, parameter_server, exploit)
        self.session = session.Session(pg_model.PolicyModel())

    def begin(self):
        self.load_shared_parameters()
        super(PGBatch, self).begin()

    def step(self, reward, state, terminal):
        if reward is not None:
            self.push_experience(reward)
        if terminal and state is not None:
            logger.warning('PGBatch.step ignores state in case of terminal.')
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

    def reset(self):
        self.episode = episode.Episode('reward', 'state', 'action')

    # Helper methods

    def push_experience(self, reward):
        assert self.last_state is not None
        assert self.last_action is not None

        self.episode.step(
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
