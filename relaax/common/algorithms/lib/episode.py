from __future__ import absolute_import
from builtins import object
import numpy as np
from . import experience


class Episode(object):
    def __init__(self, *args):
        self.keys = args
        self.experience = None

    def begin(self):
        assert self.experience is None
        self.experience = experience.Experience(*self.keys)

    def step(self, **kwargs):
        assert self.experience is not None
        self.experience.push_record(**kwargs)

    def end(self):
        assert self.experience is not None
        experience = self.experience
        self.experience = None
        return experience


class ReplayBuffer(Episode):
    def __init__(self, buffer_size, seed=None, *args):
        super(ReplayBuffer, self).__init__(*args)
        assert buffer_size > 0, 'You have to provide positive buffer size'
        self.buffer_size = buffer_size
        if seed is not None:
            np.random.seed(seed)

    def step(self, **kwargs):
        assert self.experience is not None
        self.experience.push_record(**kwargs)
        if self.experience._len > self.buffer_size:
            self.experience.del_record(self.experience._len - self.buffer_size)

    def sample(self, batch_size=1):
        idx = np.random.choice(self.experience._len, batch_size, replace=False)
        sample = {k: [] for k in self.experience._lists}
        for k, v in sample.items():
            for i in idx:
                v.append(self.experience[k][i])
        return sample
