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

    def extend(self, **kwargs):
        assert self.experience is not None
        self.keys = self.experience.push_records(**kwargs)

    def end(self):
        assert self.experience is not None
        experience = self.experience
        self.experience = None
        return experience

    @property
    def size(self):
        return len(self.experience)


class ReplayBuffer(Episode):
    def __init__(self, keys, buffer_size=1, seed=None):
        super(ReplayBuffer, self).__init__(*keys)
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
    def __init__(self, keys, shuffle=True):
        super(Dataset, self).__init__(*keys)
        self.do_shuffle = shuffle
        self._next_id = 0

    def subset(self, elements=1, stochastic=True, keys=None):
        assert self.size > 0, 'Source dataset is empty'
        cur_keys = list(self.keys)
        if keys is not None:
            cur_keys = keys
        d = Dataset(cur_keys, shuffle=stochastic)
        d.begin()
        # elements could be
        # - integer: [:elements] or random subset with the same number of elements if stochastic=True
        # - tuple of integers: elements=(begin_index, end_index)
        # - 1-D numpy array with indices: elements=array([1, 4, 2, 0, 3])

        if stochastic:
            assert isinstance(elements, int), \
                'Elements should be pass as single int for a random (stochastic=True) subset'
            idx = np.random.choice(self.size, elements, replace=False)
            d.experience._lists = _fill(self.experience, idx, keys=cur_keys)
            return d

        d.experience._lists = _fill(self.experience, elements, keys=cur_keys)
        return d

    def shuffle(self):
        indices = np.random.choice(self.size, self.size, replace=False)

        self.experience._lists = _fill(self.experience, indices)
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

        while True:
            yield self.next_batch(batch_size)
            # TODO: test with small tails (lstm version)
            # if self._next_id >= self.size:
            if self._next_id > self.size - batch_size:
                break
        self._next_id = 0


def _fill(src, idx, keys=None):
    if keys is None:
        dst = {k: [] for k in src._lists}
    else:
        dst = {k: [] for k in keys}
    if type(idx) is int:
        assert idx <= len(src), 'Requesting size of the new data is larger than the current one'
        for k, v in dst.items():
            v.extend(src[k][:idx])
    elif type(idx) is tuple:
        assert idx[1] <= len(src), 'Requesting size of the new data is larger than the current one'
        for k, v in dst.items():
            v.extend(src[k][idx[0]:idx[1]])
    else:
        assert len(idx) <= len(src), 'Requesting size of the new data is larger than the current one'
        for k, v in dst.items():
            v.extend(np.asarray(src[k])[idx])
    return dst
