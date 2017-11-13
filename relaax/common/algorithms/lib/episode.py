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

    @property
    def size(self):
        return len(self.experience)


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
        if self.size > self.buffer_size:
            self.experience.del_record(self.size - self.buffer_size)

    def sample(self, batch_size=1):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return _fill(self.experience, idx)


class Dataset(Episode):
    def __init__(self, shuffle=True, *args):
        super(Dataset, self).__init__(*args)
        self.do_shuffle = shuffle
        self._next_id = 0

    def subset(self, num_elements=1, stochastic=True, indices=None):
        assert self.size > 0, 'Source dataset is empty'
        d = Dataset(*self.keys, shuffle=stochastic)
        # num_elements could be a tuple: (begin_index, end_index)

        if indices is not None:
            assert len(indices) <= self.size, 'Requesting size of the new dataset is larger than current'
            d.experience._lists = _fill(self.experience, indices)
            return d

        assert num_elements <= self.size, 'Requesting size of the new dataset is larger than current'

        if stochastic:
            indices = np.random.choice(self.size, num_elements, replace=False)
            d.experience._lists = _fill(self.experience, indices)
            return d

        d.experience._lists = _fill(self.experience, num_elements)
        return d

    def shuffle(self):
        indices = np.random.choice(self.size, self.size, replace=False)

        for key in self.experience._lists:
            self.experience[key] = np.asarray(self.experience[key])[indices]

        self._next_id = 0

    def next_batch(self, batch_size):
        if self._next_id >= self.size and self.do_shuffle:
            self.shuffle()

        cur_id = self._next_id
        cur_batch_size = min(batch_size, self.size - self._next_id)
        self._next_id += cur_batch_size

        return _fill(self.experience, (cur_id, cur_id+cur_batch_size))

    def iterate_once(self, batch_size):
        if self.do_shuffle:
            self.shuffle()

        while self._next_id <= self.size - batch_size:
            yield self.next_batch(batch_size)
        self._next_id = 0


def _fill(src, idx):
    dst = {k: [] for k in src._lists}
    if type(idx) is int:
        for k, v in dst.items():
            v.extend(src[k][:idx])
    elif type(idx) is tuple:
        for k, v in dst.items():
            v.extend(src[k][idx[0]:idx[1]])
    else:
        for k, v in dst.items():
            v.extend(np.asarray(src[k])[idx])
    return dst
