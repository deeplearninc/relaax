import tensorflow as tf
import scipy.optimize
from keras.layers.core import Layer
from collections import OrderedDict

from .distributions import *
from .misc_utils import *
from .filters import *

dtype = tf.float32
concat = np.concatenate


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

    def call(self, x):
        Mean = x  # Mean = x * 0.1
        Std = tf.tile(tf.reshape(tf.exp(self.logstd), [1, -1]), (tf.shape(Mean)[0], 1))
        return tf.concat([Mean, Std], axis=1)


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
        return - 0.5 * tf.reduce_sum(tf.square((a - mean0) / std0), axis=1) \
               - 0.5 * tf.log(2.0 * np.pi) * self.d - tf.reduce_sum(tf.log(std0), axis=1)

    def likelihood(self, a, prob):
        return tf.exp(self.loglikelihood(a, prob))

    def kl(self, prob0, prob1):
        mean0 = prob0[:, :self.d]
        std0 = prob0[:, self.d:]
        mean1 = prob1[:, :self.d]
        std1 = prob1[:, self.d:]
        return tf.reduce_sum(tf.log(std1 / std0), axis=1) + tf.reduce_sum(
            ((tf.square(std0) + tf.square(mean0 - mean1)) / (2.0 * tf.square(std1))), axis=1) - 0.5 * self.d

    def entropy(self, prob):
        std_nd = prob[:, self.d:]
        return tf.reduce_sum(tf.log(std_nd), axis=1) + .5 * np.log(2 * np.pi * np.e) * self.d

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

def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return np.prod(var_shape(x))


def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat([tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, grads)], axis=0)


class GetFlat(object):
    def __init__(self, var_list, session):
        self.session = session
        self.op = tf.concat([tf.reshape(v, [numel(v)]) for v in var_list], axis=0)

    def __call__(self):
        return self.op.eval(session=self.session)


class SetFromFlat(object):
    def __init__(self, var_list, session):
        self.session = session

        shapes = map(var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in shapes)
        self.theta = theta = tf.placeholder(dtype, [total_size])
        start = 0
        updates = []
        # for v in var_list:
        for (shape, v) in zip(shapes, var_list):
            size = np.prod(shape)
            updates.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
            start += size
        self.op = tf.group(*updates)

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})


class EzFlat(object):
    def __init__(self, var_list, session):
        self.gf = GetFlat(var_list, session)
        self.sff = SetFromFlat(var_list, session)

    def set_params_flat(self, theta):
        self.sff(theta)

    def get_params_flat(self):
        return self.gf()


class LbfgsOptimizer(EzFlat):
    def __init__(self, session, loss,  params, symb_args, extra_losses=None, maxiter=25):
        EzFlat.__init__(self, params, session)

        self.all_losses = OrderedDict()
        self.all_losses["loss"] = loss
        if extra_losses is not None:
            self.all_losses.update(extra_losses)

        self.f_lossgrad = TensorFlowLazyFunction(list(symb_args), [loss, flatgrad(loss, params)], session)
        self.f_losses = TensorFlowLazyFunction(symb_args, list(self.all_losses.values()), session)
        self.maxiter = maxiter

    def update(self, *args):
        thprev = self.get_params_flat()

        def lossandgrad(th):
            self.set_params_flat(th)
            l, g = self.f_lossgrad(*args)
            g = g.astype('float64')
            return l, g

        losses_before = self.f_losses(*args)
        print('lossandgrad', repr(lossandgrad))
        print('thprev', repr(thprev))
        print('self.maxiter', repr(self.maxiter))
        theta, _, opt_info = scipy.optimize.fmin_l_bfgs_b(lossandgrad, thprev, maxiter=self.maxiter)
        del opt_info['grad']
        print('opt_info', opt_info)     # future

        self.set_params_flat(theta)
        losses_after = self.f_losses(*args)
        info = OrderedDict()

        for (name, lossbefore, lossafter) in zip(self.all_losses.keys(), losses_before, losses_after):
            info[name+"_before"] = lossbefore
            info[name+"_after"] = lossafter
        return info


class NnRegression(EzPickle):
    def __init__(self, net, session, mixfrac=1.0, maxiter=25):
        EzPickle.__init__(self, net, mixfrac, maxiter)
        self.net = net
        self.mixfrac = mixfrac

        x_nx = net.input
        self.predict = TensorFlowLazyFunction([x_nx], net.output, session)

        ypred_ny = net.output
        ytarg_ny = tf.placeholder(dtype, name='ytarg')

        var_list = net.trainable_weights
        l2 = 1e-3 * tf.add_n([tf.reduce_sum(tf.square(v)) for v in var_list])

        mse = tf.reduce_mean(tf.square(ytarg_ny - ypred_ny))
        symb_args = [x_nx, ytarg_ny]

        loss = mse + l2
        self.opt = LbfgsOptimizer(session, loss, var_list, symb_args,
                                  maxiter=maxiter, extra_losses={"mse": mse, "l2": l2})

    def fit(self, x_nx, ytarg_ny):
        nY = ytarg_ny.shape[1]

        ypredold_ny = self.predict(x_nx)
        out = self.opt.update(x_nx, ytarg_ny*self.mixfrac + ypredold_ny*(1-self.mixfrac))
        yprednew_ny = self.predict(x_nx)

        out["PredStdevBefore"] = ypredold_ny.std()
        out["PredStdevAfter"] = yprednew_ny.std()
        out["TargStdev"] = ytarg_ny.std()

        if nY == 1:
            out["EV_before"] = explained_variance_2d(ypredold_ny, ytarg_ny)[0]
            out["EV_after"] = explained_variance_2d(yprednew_ny, ytarg_ny)[0]
        else:
            out["EV_avg"] = explained_variance(yprednew_ny.ravel(), ytarg_ny.ravel())
        return out


class NnVf(object):
    def __init__(self, vnet, timestep_limit, regression_params, session):
        self.reg = NnRegression(vnet, session, **regression_params)
        self.timestep_limit = timestep_limit

    def predict(self, path):
        ob_no = self.preproc(path["observation"])
        return self.reg.predict(ob_no)[:, 0]

    def fit(self, paths):
        ob_no = concat([self.preproc(path["observation"]) for path in paths], axis=0)
        vtarg_n1 = concat([path["return"] for path in paths]).reshape(-1, 1)
        return self.reg.fit(ob_no, vtarg_n1)

    def preproc(self, ob_no):
        return concat([ob_no, np.arange(len(ob_no)).reshape(-1, 1) / float(self.timestep_limit)], axis=1)


# ================================================================
# Auxiliary functions
# ================================================================

class TensorFlowUpdateFunction(object):
    def __init__(self, inputs, outputs, session, updates=()):
        self._inputs = inputs
        self._outputs = outputs
        self._updates = updates
        self.session = session

    def __call__(self, *args, **kwargs):
        feeds = {}
        for (argpos, arg) in enumerate(args):
            feeds[self._inputs[argpos]] = arg

        try:
            outputs_identity = [tf.identity(output) for output in self._outputs]
            output_is_list = True
        except TypeError:
            outputs_identity = [tf.identity(self._outputs)]
            output_is_list = False

        # with tf.control_dependencies(outputs_identity):
        assign_ops = [tf.assign(variable, replacement) for variable, replacement in self._updates]

        outputs_list = self.session.run(outputs_identity + assign_ops, feeds)[:len(outputs_identity)]

        if output_is_list:
            return outputs_list
        else:
            assert len(outputs_list) == 1
        return outputs_list[0]
