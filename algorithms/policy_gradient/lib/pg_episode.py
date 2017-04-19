from relaax.server.common import session
from relaax.common.algorithms.lib import episode
from relaax.common.algorithms.lib import utils

from .. import pg_config
from .. import pg_model


class PGEpisode(object):
    def __init__(self, parameter_server, exploit):
        self.exploit = exploit
        self.ps = parameter_server
        self.session = session.Session(pg_model.PolicyModel())
        self.reset()

    @property
    def experience_size(self):
        return self.episode.experience_size

    def begin(self):
        self.load_shared_parameters()
        self.episode.begin()

    def step(self, reward, state, terminal):
        if state is None:
            action = None
        else:
            action = self.action_from_policy(state)
            assert action is not None

        self.episode.step(reward=reward, state=state, action=action)

        return action

    def end(self):
        experience = self.episode.end()
        if not exploit:
            self.apply_gradients(self.compute_gradients(experience))

    def reset(self):
        self.episode = episode.Episode('reward', 'state', 'action')

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
