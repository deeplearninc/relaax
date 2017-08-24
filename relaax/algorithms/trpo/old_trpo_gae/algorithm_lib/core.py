import tensorflow as tf
import scipy.optimize
from collections import OrderedDict

from distributions import *
from misc_utils import *
from filters import *

dtype = tf.float32
concat = np.concatenate


# ================================================================
# Stochastic policies
# ================================================================

class ProbType(object):
    def kl(self, prob0, prob1):
        raise NotImplementedError

    def entropy(self, prob):
        raise NotImplementedError

    def maxprob(self, prob):
        raise NotImplementedError


class Categorical(ProbType):
    def __init__(self, n):
        self.n = n

    def kl(self, prob0, prob1):
        return tf.reduce_sum((prob0 * tf.log(prob0 / prob1)), axis=1)

    def entropy(self, prob0):
        return -tf.reduce_sum((prob0 * tf.log(prob0)), axis=1)

    @staticmethod
    def sample(prob):
        # distributions.categorical_sample(prob)
        return categorical_sample(prob)

    def maxprob(self, prob):
        return tf.argmax(prob, axis=1)


class DiagGauss(ProbType):
    def __init__(self, d):
        self.d = d

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


class StochPolicyKeras(object):
    def __init__(self, probtype, relaax_session, relaax_metrics):
        self._probtype = probtype
        self._relaax_session = relaax_session
        self._relaax_metrics = relaax_metrics

    def act(self, ob, stochastic=True):
        prob = self._relaax_session.op_get_action(state=ob[None])
        self._relaax_metrics.histogram('action', prob)
        if stochastic:
            return self._probtype.sample(prob)[0], {"prob": prob[0]}
        else:
            return self._probtype.maxprob(prob)[0], {"prob": prob[0]}


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


class LbfgsOptimizer(object):
    def __init__(self, session, loss,  params, symb_args, extra_losses=None, maxiter=25):
        self.ezflat = EzFlat(params, session)

        self.all_losses = OrderedDict()
        self.all_losses["loss"] = loss
        if extra_losses is not None:
            self.all_losses.update(extra_losses)

        self.f_lossgrad = TensorFlowLazyFunction(symb_args, [loss, flatgrad(loss, params)], session)
        self.f_losses = TensorFlowLazyFunction(symb_args, self.all_losses.values(), session)
        self.maxiter = maxiter

    def update(self, *args):
        thprev = self.ezflat.get_params_flat()

        def lossandgrad(th):
            self.ezflat.set_params_flat(th)
            l, g = self.f_lossgrad(*args)
            g = g.astype('float64')
            return l, g

        losses_before = self.f_losses(*args)
        theta, _, opt_info = scipy.optimize.fmin_l_bfgs_b(lossandgrad, thprev, maxiter=self.maxiter)
        del opt_info['grad']
        print('opt_info', opt_info)     # future

        self.ezflat.set_params_flat(theta)
        losses_after = self.f_losses(*args)
        info = OrderedDict()

        for (name, lossbefore, lossafter) in zip(self.all_losses.keys(), losses_before, losses_after):
            info[name+"_before"] = lossbefore
            info[name+"_after"] = lossafter
        return info


class NnRegression(object):
    def __init__(self, net, session, mixfrac=1.0, maxiter=25):
        self.mixfrac = mixfrac

        x_nx = net.ph_state.node
        self.predict = TensorFlowLazyFunction([x_nx], net.value.node, session)

        ytarg_ny = tf.placeholder(dtype, name='ytarg')

        var_list = net.trainable_weights
        l2 = 1e-3 * tf.add_n([tf.reduce_sum(tf.square(v)) for v in var_list])

        mse = tf.reduce_mean(tf.square(ytarg_ny - net.value.node))
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
    def __init__(self, vnet, timestep_limit, regression_params, session, metrics):
        self.reg = NnRegression(vnet, session, **regression_params)
        self.timestep_limit = timestep_limit
        self.metrics = metrics

    def predict(self, path):
        ob_no = self.preproc(path["observation"])
        value = self.reg.predict(ob_no)[:, 0]
        self.metrics.histogram('critic', value)
        return value

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
