from __future__ import print_function

import logging
import numpy as np

from .. import trpo_config
from . import core
from . import dataset

logger = logging.getLogger(__name__)
losses_names = ['surr_loss', 'kl', 'entropy']


def make_filter(config):
    if config.use_filter:
        return core.ZFilter(core.RunningStatExt(config.input.shape), clip=5)
    return core.IDENTITY


def make_probtype():
    if trpo_config.config.output.continuous:
        return core.DiagGauss(trpo_config.config.output.action_size)
    return core.Categorical(trpo_config.config.output.action_size)


def make_policy_wrapper(relaax_session, relaax_metrics):
    return core.StochPolicy(make_probtype(), relaax_session, relaax_metrics)


def make_baseline_wrapper(value_model, relaax_metrics):
    return core.NnVf(value_model, trpo_config.config.PG_OPTIONS.timestep_limit, dict(mixfrac=0.1),
                     relaax_metrics)


class PpoUpdater(object):
    def __init__(self, policy_model):
        self.policy_model = policy_model
        if not hasattr(trpo_config.config.PPO, 'minibatch_size'):
            trpo_config.config.PPO.minibatch_size = None

        if not hasattr(trpo_config.config.PPO, 'n_epochs'):
            trpo_config.config.PPO.n_epochs = 4

    def __call__(self, paths):
        logger.debug("PPO updater called")
        prob_np = np.concatenate([path['prob'] for path in paths])
        ob_no = np.concatenate([path['observation'] for path in paths])
        state = np.reshape(ob_no, ob_no.shape + (1,))
        action_na = np.concatenate([path['action'] for path in paths])
        advantage_n = np.concatenate([path['advantage'] for path in paths])

        d = dataset.Dataset(dict(state=state, sampled_variable=action_na, adv_n=advantage_n,
                                 oldprob_np=prob_np))

        losses = []
        batch_size = trpo_config.config.PPO.minibatch_size\
            if trpo_config.config.PPO.minibatch_size is not None else state.shape[0]
        assert trpo_config.config.PPO.n_epochs > 0, 'Number of epochs should be above zero'

        for i in range(trpo_config.config.PPO.n_epochs):
            for batch in d.iterate_once(batch_size):
                new_losses, _ = self.policy_model.op_ppo_optimize(state=batch['state'],
                                                                  sampled_variable=batch['sampled_variable'],
                                                                  adv_n=batch['adv_n'],
                                                                  oldprob_np=batch['oldprob_np'])
                losses.append(new_losses)

        return dict(zip(losses_names, np.mean(losses, axis=0)))


class TrpoCalculator(object):
    def __init__(self, policy_model, paths):
        self.policy_model = policy_model
        self.paths = paths
        self.prob_np = np.concatenate([path['prob'] for path in paths])
        ob_no = np.concatenate([path['observation'] for path in paths])
        self.state = np.reshape(ob_no, ob_no.shape + (1,))
        self.action_na = np.concatenate([path['action'] for path in paths])
        self.advantage_n = np.concatenate([path['advantage'] for path in paths])

        if trpo_config.config.TRPO.linesearch_type == "Origin":
            self.search = linesearch
        elif trpo_config.config.TRPO.linesearch_type == "Adaptive":
            self.search = linesearch2
        else:
            assert False, "You have to provide search type between: Origin | Adaptive"

        # Stub for current gradient vector
        self.g = None
        self.init()

    def __call__(self):
        thprev = self.policy_model.op_get_weights_flatten()

        if np.allclose(self.g, 0):
            logger.debug('got zero gradient. not updating')
        else:
            stepdir = cg(self.fisher_vector_product, -self.g)
            shs = .5 * stepdir.dot(self.fisher_vector_product(stepdir))
            lm = np.sqrt(shs / trpo_config.config.TRPO.max_kl)
            logger.debug('lagrange multiplier: {}, gnorm: {}'.format(lm, np.linalg.norm(self.g)))
            fullstep = stepdir / lm
            neggdotstepdir = -self.g.dot(stepdir)

            def loss(th):
                self.policy_model.op_set_weights_flatten(value=th)
                # Calculates losses in order: surrogate loss, KL divergence, entropy penalty
                return self.policy_model.op_losses(state=self.state,
                                                   sampled_variable=self.action_na,
                                                   adv_n=self.advantage_n,
                                                   prob_variable=self.prob_np)

            success, theta, losses = self.search(loss, thprev, fullstep, neggdotstepdir/lm)
            print('Success: {}'.format(success))

            success, theta = linesearch(loss, thprev, fullstep, neggdotstepdir/lm)
            print('success', success)
            self.policy_model.op_set_weights_flatten(value=theta)
            return dict(zip(losses_names, np.mean(losses, axis=0)))

    def init(self):
        pass

    def fisher_vector_product(self, vec):
        pass


class TrpoD1Calculator(TrpoCalculator):
    def init(self):
        gs = []
        for path in self.paths:
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
    def init(self):
        self.g = self.policy_model.op_compute_gradient(state=self.state, sampled_variable=self.action_na,
                                                       adv_n=self.advantage_n, oldprob_np=self.prob_np)

    def fisher_vector_product(self, p):
        fvp = self.policy_model.op_fisher_vector_product(tangent=p, state=self.state,
                                                         sampled_variable=self.action_na,
                                                         adv_n=self.advantage_n, prob_variable=self.prob_np)
        return fvp + trpo_config.config.TRPO.cg_damping * p


class TrpoD1Updater(object):
    def __init__(self, policy_model):
        self.policy_model = policy_model

    def __call__(self, paths):
        return TrpoD1Calculator(self.policy_model, paths)()


class TrpoD2Updater(object):
    def __init__(self, policy_model):
        self.policy_model = policy_model

    def __call__(self, paths):
        return TrpoD2Calculator(self.policy_model, paths)()


def Updater(policy_model):
    mapping = {'ppo': PpoUpdater, 'trpo-d1': TrpoD1Updater, 'trpo-d2': TrpoD2Updater}
    return mapping[trpo_config.config.subtype](policy_model)


def linesearch(f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    """
    Backtracking linesearch, where expected_improve_rate is the slope dy/dx at the initial point
    """
    fval, kl, ent = f(x)
    losses = [(fval, kl, ent)]
    logger.debug('fval before: {}'.format(fval))
    for stepfrac in .5 ** np.arange(max_backtracks):
        xnew = x + stepfrac * fullstep
        new_fval, kl, ent = f(xnew)
        losses.append((new_fval, kl, ent))
        actual_improve = fval - new_fval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve/expected_improve
        print('a/e/r:', actual_improve, expected_improve, ratio)
        if ratio > accept_ratio and actual_improve > 0:
            logger.debug('fval after: {}'.format(new_fval))
            return True, xnew, losses
    return False, x, losses


def linesearch2(f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
    """
    Simpler version of linesearch
    """
    fval, kl, ent = f(x)
    losses = [(fval, kl, ent)]
    logger.debug('fval before: {}'.format(fval))
    for stepfrac in .5 ** np.arange(max_backtracks):
        xnew = x + stepfrac * fullstep
        new_fval, kl, ent = f(xnew)
        losses.append((new_fval, kl, ent))
        actual_improve = fval - new_fval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve/expected_improve
        print('a/e/r', actual_improve, expected_improve, ratio)
        if not np.isfinite((new_fval, kl)).all():
            logger.debug("Got non-finite value of losses -- bad!")
        elif kl > trpo_config.config.TRPO.max_kl * 1.5:
            logger.debug("violated KL constraint. shrinking step.")
        elif actual_improve < 0:
            logger.debug("surrogate didn't improve. shrinking step.")
        else:
            logger.debug("Stepsize OK!")
            break
    else:
        logger.debug("couldn't compute a good step")
        return False, x, losses

    return True, xnew, losses


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
