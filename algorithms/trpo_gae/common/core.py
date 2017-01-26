from keras.layers.core import Layer


class ConcatFixedStd(Layer):
    input_ndim = 2


class ProbType(object):
    pass


class Categorical(ProbType):
    def __init__(self, n):
        self.n = n


class DiagGauss(ProbType):
    def __init__(self, d):
        self.d = d
