from relaax.common.algorithms.lib import episode

import pg_base_episode
from .. import pg_config


class PGEpisode(pg_base_episode.PGBaseEpisode):
    def __init__(self, parameter_server):
        super(PGEpisode, self).__init__(parameter_server)
        self.reset()

    def update(self, reward, state, terminal):
        assert (state is None) == terminal

        if not self.episode.in_episode:
            self.load_shared_parameters()
            self.episode.begin()

        if state is None:
            action = None
        else:
            action = self.action_from_policy(state)
            assert action is not None

        self.episode.step(reward=reward, state=state, action=action)
        if (self.episode.size == pg_config.config.batch_size) or terminal:
            experience = self.episode.end()
            self.apply_gradients(self.compute_gradients(experience))

        return action

    def reset(self):
        self.episode = episode.Episode('reward', 'state', 'action')
        return True
