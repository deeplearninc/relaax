import numpy as np


class ZFilter(object):
    """ y = (x-mean)/std
    using running estimates of mean, std """
    def __init__(self, shape_or_object, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = shape_or_object
        if type(shape_or_object) is tuple:
            self.rs = RunningStat(shape_or_object)

    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        if self.demean:
            x -= self.rs.mean
        if self.destd:
            x /= (self.rs.std+1e-8)
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
        assert x.shape == self._M.shape
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
        return self._S/(self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class RunningStatExt(RunningStat):
    def __init__(self, shape):
        super(RunningStatExt, self).__init__(shape)
        self._inN = None
        self._inM = None
        self._inS = None

    def set(self, n, M, S):
        self._n, self._inN = n, n
        assert M.shape == self._M.shape
        self._M = M.copy()
        self._S = S.copy()
        self._inM = M.copy()
        self._inS = S.copy()

    def get_diff(self):
        diffM = self._M*self._n - self._inM*self._inN
        diffS = self._S - self._inS
        return diffM, diffS
