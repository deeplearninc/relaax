from __future__ import absolute_import

import numpy as np
import tensorflow as tf
import scipy.optimize

from . import distributions
from . import misc_utils


# ================================================================
# Stochastic policies
# ================================================================

class ProbType(object):
    def entropy(self, prob):
        raise NotImplementedError

    def maxprob(self, prob):
        raise NotImplementedError


class Categorical(ProbType):
    def __init__(self, n):
        self.n = n

    @staticmethod
    def sample(prob):
        return distributions.categorical_sample(prob)

    def maxprob(self, prob):
        return tf.argmax(prob, axis=1)


class DiagGauss(ProbType):
    def __init__(self, d):
        self.d = d

    def sample(self, prob):
        mean_nd = prob[:, :self.d]
        std_nd = prob[:, self.d:]
        return np.random.randn(prob.shape[0], self.d).astype(np.float32) * std_nd + mean_nd

    def maxprob(self, prob):
        return prob[:, :self.d]


class StochPolicy(object):
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

class LbfgsOptimizer(object):
    def __init__(self, value_model, maxiter=25):
        self.value_model = value_model
        self.maxiter = maxiter

    def update(self, state, ytarg_ny):
        thprev = self.value_model.op_get_weights_flatten()

        def lossandgrad(th):
            self.value_model.op_set_weights_flatten(value=th)
            l, g = self.value_model.op_compute_loss_and_gradient(state=state, ytarg_ny=ytarg_ny)
            return l, g.astype('float64')

        losses_before = self.value_model.op_losses(state=state, ytarg_ny=ytarg_ny)
        theta, _, opt_info = scipy.optimize.fmin_l_bfgs_b(lossandgrad, thprev, maxiter=self.maxiter)
        del opt_info['grad']
        print('opt_info', opt_info)     # future

        self.value_model.op_set_weights_flatten(value=theta)
        losses_after = self.value_model.op_losses(state=state, ytarg_ny=ytarg_ny)
        info = {}

        for (name, lossbefore, lossafter) in zip(["loss", "mse", "l2"], losses_before, losses_after):
            info[name+"_before"] = lossbefore
            info[name+"_after"] = lossafter
        return info


class NnRegression(object):
    def __init__(self, value_model, mixfrac=1.0, maxiter=25):
        self.value_model = value_model
        self.mixfrac = mixfrac

        self.opt = LbfgsOptimizer(value_model, maxiter=maxiter)

    def fit(self, x_nx, ytarg_ny):
        nY = ytarg_ny.shape[1]

        ypredold_ny = self.predict(x_nx)
        out = self.opt.update(x_nx, ytarg_ny*self.mixfrac + ypredold_ny*(1-self.mixfrac))
        yprednew_ny = self.predict(x_nx)

        out["PredStdevBefore"] = ypredold_ny.std()
        out["PredStdevAfter"] = yprednew_ny.std()
        out["TargStdev"] = ytarg_ny.std()

        if nY == 1:
            out["EV_before"] = misc_utils.explained_variance_2d(ypredold_ny, ytarg_ny)[0]
            out["EV_after"] = misc_utils.explained_variance_2d(yprednew_ny, ytarg_ny)[0]
        else:
            out["EV_avg"] = misc_utils.explained_variance(yprednew_ny.ravel(), ytarg_ny.ravel())
        return out

    def predict(self, x):
        return self.value_model.op_value(state=x)


class NnVf(object):
    def __init__(self, value_model, timestep_limit, regression_params, metrics):
        self.reg = NnRegression(value_model, **regression_params)
        self.timestep_limit = timestep_limit
        self.metrics = metrics

    def predict(self, path):
        ob_no = self.preproc(path["observation"])
        value = self.reg.predict(ob_no)[:, 0]
        self.metrics.histogram('critic', value)
        return value

    def fit(self, paths):
        ob_no = np.concatenate([self.preproc(path["observation"]) for path in paths], axis=0)
        vtarg_n1 = np.concatenate([path["return"] for path in paths]).reshape(-1, 1)
        return self.reg.fit(ob_no, vtarg_n1)

    def preproc(self, ob_no):
        length = len(ob_no)
        if ob_no[0].ndim > 1:
            ob_no = [obs.flatten() for obs in ob_no]
        return np.concatenate([ob_no, np.arange(length).reshape(-1, 1) / float(self.timestep_limit)],
                              axis=1)
