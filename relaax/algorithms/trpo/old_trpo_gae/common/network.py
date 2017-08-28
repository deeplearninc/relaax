from __future__ import print_function

import numpy as np
import tensorflow as tf

from ... import trpo_config

from relaax.algorithms.trpo.old_trpo_gae.algorithm_lib import core


D2 = False


def make_filters(config):
    if config.use_filter:
        obfilter = core.ZFilter(core.RunningStatExt(config.input.shape), clip=5)
        rewfilter = core.ZFilter((), demean=False, clip=10)
    else:
        obfilter = core.IDENTITY
        rewfilter = core.IDENTITY
    return obfilter, rewfilter


def make_probtype():
    if trpo_config.config.output.continuous:
        return core.DiagGauss(trpo_config.config.output.action_size)
    return core.Categorical(trpo_config.config.output.action_size)


def make_policy_wrapper(relaax_session, relaax_metrics):
    return core.StochPolicyKeras(make_probtype(), relaax_session, relaax_metrics)


def make_baseline_wrapper(relaax_session, relaax_metrics):
    return core.NnVf(relaax_session.model.value_net, trpo_config.config.PG_OPTIONS.timestep_limit,
                     dict(mixfrac=0.1), relaax_session.session, relaax_metrics)


class TrpoUpdater(object):
    def __init__(self, relaax_session):
        probtype = make_probtype()
        net = relaax_session.model.policy_net

        self.relaax_session = relaax_session

        params = net.trainable_weights

        ob_no = net.ph_state.node
        act_na = net.ph_sampled_variable.node
        adv_n = net.ph_adv_n.node

        # Probability distribution:
        prob_np = net.actor.node
        oldprob_np = net.ph_prob_variable.node

        args = [ob_no, act_na, adv_n, oldprob_np]

        ent = tf.reduce_mean(probtype.entropy(prob_np))

        self.compute_losses = core.TensorFlowLazyFunction(args, [net.surr.node, net.kl.node, ent],
                                                          relaax_session.session)

        tangent = tf.placeholder(core.dtype, name='flat_tan')

        if D2:
            grads1 = tf.gradients(net.kl_first_fixed.node, params)
            gvp = []
            start = 0
            for g in grads1:
                size = np.prod(g.shape.as_list())
                gvp.append(tf.reduce_sum(tf.reshape(g, [-1]) * tangent[start:start + size]))
                start += size

            grads2 = tf.gradients(gvp, params)
            fvp = tf.concat([tf.reshape(g, [-1]) for g in grads2], axis=0)

            self.compute_fisher_vector_product = core.TensorFlowLazyFunction([tangent] + args, fvp,
                                                                             relaax_session.session)
        else:
            def cfvp(tangent, g):
                v = np.matmul(g, tangent)
                return np.mean(np.multiply(g, np.expand_dims(v, -1)), axis=0)

            self.cfvp = cfvp

    def __call__(self, paths):
        prob_np = core.concat([path["prob"] for path in paths])
        ob_no = core.concat([path["observation"] for path in paths])
        action_na = core.concat([path["action"] for path in paths])
        advantage_n = core.concat([path["advantage"] for path in paths])
        args = (np.reshape(ob_no, ob_no.shape + (1,)), action_na, advantage_n, prob_np)

        thprev = self.relaax_session.op_get_weights_flatten()

        if D2:
            g = self.relaax_session.op_compute_policy_gradient(state=np.reshape(ob_no, ob_no.shape + (1,)),
                                                               sampled_variable=action_na, adv_n=advantage_n,
                                                               oldprob_np=prob_np)

            def fisher_vector_product(p):
                return self.compute_fisher_vector_product(p, *args) + trpo_config.config.TRPO.cg_damping * p
        else:
            gs = []
            for path in paths:
                prob_np = path["prob"]
                ob_no = np.asarray(path["observation"])
                action_na = path["action"]
                advantage_n = path["advantage"]

                g = self.relaax_session.op_compute_policy_gradient(state=np.reshape(ob_no,
                                                                   ob_no.shape + (1,)),
                                                                   sampled_variable=action_na,
                                                                   adv_n=advantage_n, oldprob_np=prob_np)
                gs.append(g)
            g_matrix = np.stack(gs)

            def fisher_vector_product(p):
                return self.cfvp(p, g_matrix) + trpo_config.config.TRPO.cg_damping * p

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
                self.relaax_session.op_set_weights_flatten(value=th)
                return self.compute_losses(*args)[0]

            success, theta = linesearch(loss, thprev, fullstep, neggdotstepdir/lm)
            print("success", success)
            self.relaax_session.op_set_weights_flatten(value=theta)


def linesearch(f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    """
    Backtracking linesearch, where expected_improve_rate is the slope dy/dx at the initial point
    """
    fval = f(x)
    print("fval before", fval)
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
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
