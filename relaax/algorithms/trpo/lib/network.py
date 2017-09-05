from __future__ import print_function

import numpy as np

from .. import trpo_config
from . import core


PPO = False
D2 = True


def make_filter(config):
    if config.use_filter:
        return core.ZFilter(core.RunningStatExt(config.input.shape), clip=5)
    return core.IDENTITY


def make_probtype():
    if trpo_config.config.output.continuous:
        return core.DiagGauss(trpo_config.config.output.action_size)
    return core.Categorical(trpo_config.config.output.action_size)


def make_policy_wrapper(relaax_session, relaax_metrics):
    return core.StochPolicyKeras(make_probtype(), relaax_session, relaax_metrics)


def make_baseline_wrapper(value_model, relaax_metrics):
    return core.NnVf(value_model, trpo_config.config.PG_OPTIONS.timestep_limit, dict(mixfrac=0.1),
                     relaax_metrics)


class PpoUpdater(object):
    def __init__(self, policy_model):
        self.policy_model = policy_model

    def __call__(self, paths):
        prob_np = np.concatenate([path['prob'] for path in paths])
        ob_no = np.concatenate([path['observation'] for path in paths])
        state = np.reshape(ob_no, ob_no.shape + (1,))
        action_na = np.concatenate([path['action'] for path in paths])
        advantage_n = np.concatenate([path['advantage'] for path in paths])

        for i in range(trpo_config.config.PPO.n_epochs):
            self.policy_model.op_ppo_optimize(state=state, sampled_variable=action_na, adv_n=advantage_n,
                                              oldprob_np=prob_np)
        # TODO: add KL new/old for debug


class TrpoCalculator(object):
    def __init__(self, policy_model, paths):
        self.policy_model = policy_model
        self.prob_np = np.concatenate([path['prob'] for path in paths])
        ob_no = np.concatenate([path['observation'] for path in paths])
        self.state = np.reshape(ob_no, ob_no.shape + (1,))
        self.action_na = np.concatenate([path['action'] for path in paths])
        self.advantage_n = np.concatenate([path['advantage'] for path in paths])
        self.init(paths)

    def __call__(self):
        thprev = self.policy_model.op_get_weights_flatten()

        if np.allclose(self.g, 0):
            print('got zero gradient. not updating')
        else:
            stepdir = cg(self.fisher_vector_product, -self.g)
            shs = .5 * stepdir.dot(self.fisher_vector_product(stepdir))
            lm = np.sqrt(shs / trpo_config.config.TRPO.max_kl)
            print('lagrange multiplier:', lm, 'gnorm:', np.linalg.norm(self.g))
            fullstep = stepdir / lm
            neggdotstepdir = -self.g.dot(stepdir)

            def loss(th):
                self.policy_model.op_set_weights_flatten(value=th)
                surr, kl, ent = self.policy_model.op_losses(state=self.state,
                                                            sampled_variable=self.action_na,
                                                            adv_n=self.advantage_n,
                                                            prob_variable=self.prob_np)
                return surr

            success, theta = linesearch(loss, thprev, fullstep, neggdotstepdir/lm)
            print('success', success)
            self.policy_model.op_set_weights_flatten(value=theta)


class TrpoD1Calculator(TrpoCalculator):
    def init(self, paths):
        gs = []
        for path in paths:
            ob_no = np.asarray(path['observation'])
            state = np.reshape(ob_no, ob_no.shape + (1,))
            g = self.policy_model.op_compute_gradient(state=state, sampled_variable=path['action'],
                                                      adv_n=path['advantage'], oldprob_np=path['prob'])
            gs.append(g)
        self.g_matrix = np.stack(gs)
        self.g = np.mean(self.g_matrix, axis=0)

    def fisher_vector_product(self, p):
        v = np.matmul(self.g_matrix, p)
        cfvp = np.mean(np.multiply(self.g_matrix, np.expand_dims(v, -1)), axis=0)
        return cfvp + trpo_config.config.TRPO.cg_damping * p


class TrpoD2Calculator(TrpoCalculator):
    def init(self, paths):
        self.g = self.policy_model.op_compute_gradient(state=self.state, sampled_variable=self.action_na,
                                                       adv_n=self.advantage_n, oldprob_np=self.prob_np)

    def fisher_vector_product(self, p):
        fvp = self.policy_model.op_fisher_vector_product(tangent=p, state=self.state,
                                                         sampled_variable=self.action_na,
                                                         adv_n=self.advantage_n, prob_variable=self.prob_np)
        return fvp + trpo_config.config.TRPO.cg_damping * p


class TrpoUpdater(object):
    def __init__(self, policy_model, TrpoCalculator):
        self.policy_model = policy_model
        self.TrpoCalculator = TrpoCalculator

    def __call__(self, paths):
        self.TrpoCalculator(self.policy_model, paths)()


def Updater(policy_model):
    if PPO:
        return PpoUpdater(policy_model)
    return TrpoUpdater(policy_model, TrpoD2Calculator if D2 else TrpoD1Calculator)


def linesearch(f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    """
    Backtracking linesearch, where expected_improve_rate is the slope dy/dx at the initial point
    """
    fval = f(x)
    print('fval before', fval)
    for stepfrac in .5 ** np.arange(max_backtracks):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve/expected_improve
        print('a/e/r', actual_improve, expected_improve, ratio)
        if ratio > accept_ratio and actual_improve > 0:
            print('fval after', newfval)
            return True, xnew
    return False, x


def cg(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """ Demmel p 312 """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr = '%10i %10.3g %10.3g'
    titlestr = '%10s %10s %10s'
    if verbose:
        print(titlestr % ('iter', 'residual norm', 'soln norm'))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose:
            print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose:
        print(fmtstr % (i+1, rdotr, np.linalg.norm(x)))
    return x
