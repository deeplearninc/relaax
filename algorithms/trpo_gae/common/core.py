import tensorflow as tf
from keras.layers.core import Layer

from distributions import *
from misc_utils import *
dtype = tf.float32


# ================================================================
# Keras
# ================================================================

class ConcatFixedStd(Layer):
    input_ndim = 2

    def __init__(self, **kwargs):
        Layer.__init__(self, **kwargs)
        self.logstd = None

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.logstd = tf.Variable(tf.zeros(input_dim, dtype), name='{}_logstd'.format(self.name))
        self.trainable_weights = [self.logstd]

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1] * 2

    def call(self, x, mask):
        Mean = x  # Mean = x * 0.1
        Std = tf.tile(tf.reshape(tf.exp(self.logstd), [1, -1]), (tf.shape(Mean)[0], 1))
        return tf.concat(1, [Mean, Std])


# ================================================================
# Stochastic policies
# ================================================================

class ProbType(object):
    def sampled_variable(self):
        raise NotImplementedError

    def prob_variable(self):
        raise NotImplementedError

    def likelihood(self, a, prob):
        raise NotImplementedError

    def loglikelihood(self, a, prob):
        raise NotImplementedError

    def kl(self, prob0, prob1):
        raise NotImplementedError

    def entropy(self, prob):
        raise NotImplementedError

    def maxprob(self, prob):
        raise NotImplementedError


class Categorical(ProbType):
    def __init__(self, n):
        self.n = n

    def sampled_variable(self):
        return tf.placeholder(tf.int32, name='a')

    def prob_variable(self):
        return tf.placeholder(dtype, name='prob')

    def likelihood(self, a, prob):
        return prob[tf.range(prob.shape[0]), a]

    def loglikelihood(self, a, prob):
        return tf.log(self.likelihood(a, prob))

    def kl(self, prob0, prob1):
        return (prob0 * tf.log(prob0 / prob1)).sum(axis=1)

    def entropy(self, prob0):
        return - (prob0 * tf.log(prob0)).sum(axis=1)

    @staticmethod
    def sample(prob):
        # distributions.categorical_sample(prob)
        return categorical_sample(prob)

    def maxprob(self, prob):
        return prob.argmax(axis=1)


class DiagGauss(ProbType):
    def __init__(self, d):
        self.d = d

    def sampled_variable(self):
        return tf.placeholder(dtype, name='a')

    def prob_variable(self):
        return tf.placeholder(dtype, name='prob')

    def loglikelihood(self, a, prob):
        mean0 = prob[:, :self.d]
        std0 = prob[:, self.d:]
        return - 0.5 * tf.reduce_sum(tf.square((a - mean0) / std0), 1) \
               - 0.5 * tf.log(2.0 * np.pi) * self.d - tf.reduce_sum(tf.log(std0), 1)

    def likelihood(self, a, prob):
        return tf.exp(self.loglikelihood(a, prob))

    def kl(self, prob0, prob1):
        mean0 = prob0[:, :self.d]
        std0 = prob0[:, self.d:]
        mean1 = prob1[:, :self.d]
        std1 = prob1[:, self.d:]
        return tf.reduce_sum(tf.log(std1 / std0), 1) + tf.reduce_sum(
            ((tf.square(std0) + tf.square(mean0 - mean1)) / (2.0 * tf.square(std1))), 1) - 0.5 * self.d

    def entropy(self, prob):
        std_nd = prob[:, self.d:]
        return tf.reduce_sum(tf.log(std_nd), 1) + .5 * np.log(2 * np.pi * np.e) * self.d

    def sample(self, prob):
        mean_nd = prob[:, :self.d]
        std_nd = prob[:, self.d:]
        return np.random.randn(prob.shape[0], self.d).astype(np.float32) * std_nd + mean_nd

    def maxprob(self, prob):
        return prob[:, :self.d]


class StochPolicy(object):
    @property
    def probtype(self):
        raise NotImplementedError

    @property
    def trainable_variables(self):
        raise NotImplementedError

    @property
    def input(self):
        raise NotImplementedError

    def get_output(self):
        raise NotImplementedError

    def act(self, ob, stochastic=True):
        prob = self._act_prob(ob[None])
        if stochastic:
            return self.probtype.sample(prob)[0], {"prob": prob[0]}
        else:
            return self.probtype.maxprob(prob)[0], {"prob": prob[0]}

    def finalize(self, session):
        # misc_utils.TensorFlowLazyFunction
        self._act_prob = TensorFlowLazyFunction([self.input], self.get_output(), session)


class StochPolicyKeras(StochPolicy, EzPickle):
    def __init__(self, net, probtype, session):
        # misc_utils.EzPickle
        EzPickle.__init__(self, net, probtype)
        self._net = net
        self._probtype = probtype
        self.finalize(session)

    @property
    def probtype(self):
        return self._probtype

    @property
    def net(self):
        return self._net

    @property
    def trainable_variables(self):
        return self._net.trainable_weights

    @property
    def variables(self):
        return self._net.get_params()[0]

    @property
    def input(self):
        return self._net.input

    def get_output(self):
        return self._net.output

    def get_updates(self):
        return self._net.updates

    def get_flat(self):
        return flatten(self.net.get_weights())

    def set_from_flat(self, th):
        weights = self.net.get_weights()
        self._weight_shapes = [weight.shape for weight in weights]
        self.net.set_weights(unflatten(th, self._weight_shapes))


# ================================================================
# Value functions
# ================================================================

class NnVf(object):
    def __init__(self, vnet, timestep_limit, regression_params, session):
        pass
