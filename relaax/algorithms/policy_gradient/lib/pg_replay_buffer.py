from builtins import object

from relaax.common.algorithms.lib import experience
from .. import pg_config


class PGReplayBuffer(object):
    def __init__(self, callback):
        self.callback = callback

        self.keys = ['reward', 'state', 'action']
        self.experience = None
        self._begin()

    def step(self, terminal, **kwargs):
        assert self.experience is not None
        self.experience.push_record(**kwargs)

        if (len(self.experience) == pg_config.config.batch_size) or terminal:
            self._end()
            if terminal:
                self._reset()
            self._begin()

    def _begin(self):
        assert self.experience is None
        self.callback.begin()
        self.experience = experience.Experience(*self.keys)

    def _end(self):
        assert self.experience is not None
        experience = self.experience
        self.experience = None
        self.callback.end(experience)

    def _reset(self):
        self.keys = ['reward', 'state', 'action']
        self.experience = None
        self.callback.reset()
