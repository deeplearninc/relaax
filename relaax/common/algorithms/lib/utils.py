from builtins import next
from builtins import range
from builtins import object
import logging
import numpy as np
import scipy.signal
import tensorflow as tf

from relaax.common.python.config.loaded_config import options


log = logging.getLogger(__name__)


def discounted_reward(rewards, gamma, normalize=False):
    # take 1D float array of rewards and compute discounted reward
    rewards = np.vstack(rewards)
    discounted_reward = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(rewards.size)):
        running_add = running_add * gamma + rewards[t]
        discounted_reward[t] = running_add

    # size the rewards to be unit normal
    # it helps control the gradient estimator variance
    if normalize:
        discounted_reward -= np.mean(discounted_reward)
        discounted_reward /= np.std(discounted_reward) + 1e-20

    return discounted_reward


def choose_action_descrete(probabilities, exploit=False):
    if exploit:
        return np.argmax(probabilities)   # need to set greedily param
    return np.random.choice(len(probabilities), p=probabilities)


def choose_action_continuous(mu, sigma2, min_clip=-float('inf'), max_clip=float('inf'), exploit=False):
    if min_clip is None or (type(min_clip) is list and len(min_clip) == 0):
        min_clip = -float('inf')
    if max_clip is None or (type(max_clip) is list and len(max_clip) == 0):
        max_clip = float('inf')
    if exploit:
        act, = mu
    else:
        act, = np.random.randn(*sigma2.shape).astype(np.float32) * sigma2 + mu
    return np.clip(act, min_clip, max_clip)


def assemble_and_show_graphs(*graphs):
    for graph in graphs:
        graph()
    log_dir = options.get("agent/log_dir", "log")
    log.info(('Writing TF summary to %s. '
              'Please use tensorboad to watch.') % log_dir)
    tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())


class OUNoise:
    """Ornstein-Uhlenbeck Noise Process"""
    def __init__(self, action_size, mu=0.0, theta=0.15, sigma=0.3, seed=123):
        self.action_size = action_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_size) * self.mu
        np.random.seed(seed)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = np.ones(self.action_size) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class Utils(object):
    @staticmethod
    def map(v, mapping):

        def map_(v):
            if isinstance(v, (tuple, list)):
                return [map_(v1) for v1 in v]
            if isinstance(v, dict):
                return {k: map_(v1) for k, v1 in v.items()}
            return mapping(v)

        return map_(v)

    @classmethod
    def flatten(cls, v):
        if isinstance(v, (tuple, list)):
            for vv in v:
                for vvv in cls.flatten(vv):
                    yield vvv
        elif isinstance(v, dict):
            for vv in v.values():
                for vvv in cls.flatten(vv):
                    yield vvv
        else:
            yield v

    @classmethod
    def reconstruct(cls, v, pattern):
        i = iter(v)
        result = cls.map(pattern, lambda v: next(i))
        try:
            next(i)
            assert False
        except StopIteration:
            pass
        return result

    @classmethod
    def izip(cls, v1, v2):
        if isinstance(v1, (tuple, list)):
            assert isinstance(v2, (tuple, list))
            assert len(v1) == len(v2)
            for vv1, vv2 in zip(v1, v2):
                for vvv1, vvv2 in cls.izip(vv1, vv2):
                    yield vvv1, vvv2
        elif isinstance(v1, dict):
            assert isinstance(v2, dict)
            assert len(v1) == len(v2)
            for k1, vv1 in v1.items():
                vv2 = v2[k1]
                for vvv1, vvv2 in cls.izip(vv1, vv2):
                    yield vvv1, vvv2
        else:
            yield v1, v2


class ZFilter(object):
    """ y = (x-mean)/std
    using running estimates of mean, std """
    def __init__(self, shape, demean=True, destd=True, clip=5.0, epsilon=1e-8):
        self.demean = demean
        self.destd = destd
        self.clip = clip
        self.epsilon = epsilon

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + self.epsilon)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x

    @staticmethod
    def output_shape(input_space):
        return input_space.shape


class RunningStat(object):
    # http://www.johndcook.com/blog/standard_deviation/
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape  # print(x.shape, self._M.shape)
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM)/self._n
            self._S[...] = self._S + (x - oldM)*(x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]
