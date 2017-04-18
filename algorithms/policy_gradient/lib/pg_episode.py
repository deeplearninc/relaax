from relaax.common.algorithms.lib import episode

import pg_base_episode
from .. import pg_config


class PGEpisode(pg_base_episode.PGBaseEpisode):
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
        self.apply_gradients(self.compute_gradients(experience))

    def reset(self):
        self.episode = episode.Episode('reward', 'state', 'action')
