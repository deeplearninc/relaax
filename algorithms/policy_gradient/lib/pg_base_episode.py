from relaax.server.common import session
from relaax.common.algorithms.lib import utils

from .. import pg_config
from .. import pg_model


class PGBaseEpisode(object):
    def __init__(self, parameter_server):
        self.ps = parameter_server
        self.session = session.Session(pg_model.PolicyModel())

    def update(self, reward, state, terminal):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    # Helper methods

    # reload policy weights from PS
    def load_shared_parameters(self):
        self.session.op_assign_weights(values=self.ps.op_get_weights())

    def action_from_policy(self, state):
        assert state is not None

        probabilities, = self.session.op_get_action(state=[state])
        return utils.choose_action(probabilities)

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

    def apply_gradients(self, gradients):
        self.ps.op_apply_gradients(gradients=gradients)
