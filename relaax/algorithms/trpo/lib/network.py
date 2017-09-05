from __future__ import print_function

import numpy as np
import tensorflow as tf

from .. import trpo_config

from . import core


PPO = True
D2 = False



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


class TrpoUpdater(object):
    def __init__(self, policy_model):
        self.policy_model = policy_model

    def __call__(self, paths):
        prob_np = np.concatenate([path["prob"] for path in paths])
        ob_no = np.concatenate([path["observation"] for path in paths])
        state = np.reshape(ob_no, ob_no.shape + (1,))
        action_na = np.concatenate([path["action"] for path in paths])
        advantage_n = np.concatenate([path["advantage"] for path in paths])

        thprev = self.policy_model.op_get_weights_flatten()

        if PPO:
             for i in range(trpo_config.config.PPO.n_epochs):
                 self.policy_model.op_ppo_optimize(state=state, sampled_variable=action_na,
                                                   adv_n=advantage_n, oldprob_np=prob_np)
                 # TODO: add KL new/old for debug
        else:
            if D2:
                def fisher_vector_product(p):
                    fvp = self.policy_model.op_fisher_vector_product(tangent=p, state=state,
                                                                     sampled_variable=action_na,
                                                                     adv_n=advantage_n, prob_variable=prob_np)
                    return fvp + trpo_config.config.TRPO.cg_damping * p

                g = self.policy_model.op_compute_gradient(state=state, sampled_variable=action_na,
                                                          adv_n=advantage_n, oldprob_np=prob_np)
            else:
                gs = []
                for path in paths:
                    ob_no = np.asarray(path["observation"])
                    state_ = np.reshape(ob_no, ob_no.shape + (1,))
                    g = self.policy_model.op_compute_gradient(state=state_, sampled_variable=path['action'],
                                                              adv_n=path['advantage'], oldprob_np=path['prob'])
                    gs.append(g)
                g_matrix = np.stack(gs)

                def cfvp(tangent, g):
                    v = np.matmul(g, tangent)
                    return np.mean(np.multiply(g, np.expand_dims(v, -1)), axis=0)

                def fisher_vector_product(p):
                    return cfvp(p, g_matrix) + trpo_config.config.TRPO.cg_damping * p

                g = np.mean(g_matrix, axis=0)

            if np.allclose(g, 0):
                print("got zero gradient. not updating")
            else:
                stepdir = cg(fisher_vector_product, -g)
                shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / trpo_config.config.TRPO.max_kl)
                print("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                fullstep = stepdir / lm
                neggdotstepdir = -g.dot(stepdir)

                def loss(th):
                    self.policy_model.op_set_weights_flatten(value=th)
                    surr, kl, ent = self.policy_model.op_losses(state=state, sampled_variable=action_na,
                                                          adv_n=advantage_n, prob_variable=prob_np)
                    return surr

                success, theta = linesearch(loss, thprev, fullstep, neggdotstepdir/lm)
                print("success", success)
                self.policy_model.op_set_weights_flatten(value=theta)


def linesearch(f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    """
    Backtracking linesearch, where expected_improve_rate is the slope dy/dx at the initial point
    """
    fval = f(x)
    print("fval before", fval)
    for stepfrac in .5**np.arange(max_backtracks):
        xnew = x + stepfrac*fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate*stepfrac
        ratio = actual_improve/expected_improve
        print("a/e/r", actual_improve, expected_improve, ratio)
        if ratio > accept_ratio and actual_improve > 0:
            print("fval after", newfval)
            return True, xnew
    return False, x


def cg(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """ Demmel p 312 """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    if verbose:
        print(titlestr % ("iter", "residual norm", "soln norm"))

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
