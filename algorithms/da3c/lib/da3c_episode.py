from relaax.server.common import session
from relaax.common.algorithms.lib import episode
from relaax.common.algorithms.lib import utils

from .. import da3c_config
from .. import da3c_model


class DA3CEpisode(object):
    def __init__(self, parameter_server, exploit):
        self.exploit = exploit
        self.ps = parameter_server
        self.session = session.Session(da3c_model.AgentModel())
        self.reset()
        self.last_state = None
        self.last_action = None

    @property
    def experience(self):
        return self.episode.experience

    def begin(self):
        self.load_shared_parameters()
        self.episode.begin()

    def step(self, reward, state, terminal):
        if reward is not None:
            self.push_experience(reward)
        assert (state is None) == terminal
        return self.keep_state_and_action(state)

    def end(self):
        experience = self.episode.end()
        if not self.exploit:
            self.apply_gradients(self.compute_gradients(experience))

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

    def keep_state_and_action(self, state):
        assert self.last_state is None
        assert self.last_action is None

        self.last_state = state
        if state is None:
            self.last_action = None
        else:
            self.last_action = self.action_from_policy(state)
            assert self.last_action is not None
        return self.last_action
