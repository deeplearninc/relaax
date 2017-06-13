from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Lambda

from relaax.algorithms.trpo.old_trpo_gae.algorithm_lib.core import *


def make_mlps(config):
    policy_net = Sequential()
    for (i, layeroutsize) in enumerate(config.hidden_sizes):
        input_shape = dict(input_shape=config.input.shape) if i == 0 else {}
        policy_net.add(Dense(layeroutsize, activation=config.activation, **input_shape))
    if config.output.continuous:
        policy_net.add(Dense(config.output.action_size))
        # policy_net.add(Lambda(lambda x: x * 0.1))
        policy_net.add(ConcatFixedStd())
    else:
        policy_net.add(Dense(config.output.action_size, activation="softmax"))
        # policy_net.add(Lambda(lambda x: x * 0.1))

    value_net = Sequential()
    for (i, layeroutsize) in enumerate(config.hidden_sizes):
        # add one extra feature for timestep
        input_shape = dict(input_shape=(config.input.shape[0]+1,)) if i == 0 else {}
        value_net.add(Dense(layeroutsize, activation=config.activation, **input_shape))
    value_net.add(Dense(1))

    return policy_net, value_net


def make_filters(config):
    if config.use_filter:
        obfilter = ZFilter(RunningStatExt(config.input.shape), clip=5)
        rewfilter = ZFilter((), demean=False, clip=10)
    else:
        obfilter = IDENTITY
        rewfilter = IDENTITY
    return obfilter, rewfilter


def make_wrappers(config, policy_net, value_net, session):

    if config.output.continuous:
        probtype = DiagGauss(config.output.action_size)
    else:
        probtype = Categorical(config.output.action_size)

    policy = StochPolicyKeras(policy_net, probtype, session)
    baseline = NnVf(value_net, config.PG_OPTIONS.timestep_limit, dict(mixfrac=0.1), session)

    return policy, baseline


class TrpoUpdater(object):
    def __init__(self, config, stochpol, session):
        self.cfg = config
        self.stochpol = stochpol

        probtype = stochpol.probtype
        params = stochpol.trainable_variables
        self.ezflat = EzFlat(params, session)

        ob_no = stochpol.input
        act_na = probtype.sampled_variable()
        adv_n = tf.placeholder(dtype, name='adv_n')

        # Probability distribution:
        prob_np = stochpol.get_output()
        oldprob_np = probtype.prob_variable()

        logp_n = probtype.loglikelihood(act_na, prob_np)
        oldlogp_n = probtype.loglikelihood(act_na, oldprob_np)

        # Policy gradient:
        surr = -tf.reduce_mean(tf.multiply(tf.exp(logp_n - oldlogp_n), adv_n))
        pg = flatgrad(surr, params)

        N = tf.cast(tf.shape(ob_no)[0], tf.float32)
        prob_np_fixed = tf.stop_gradient(prob_np)
        kl_firstfixed = tf.reduce_sum(probtype.kl(prob_np_fixed, prob_np)) / N

        grads = tf.gradients(kl_firstfixed, params)
        flat_tangent = tf.placeholder(dtype, name='flat_tan')

        shapes = map(var_shape, params)
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            tangents.append(tf.reshape(flat_tangent[start:(start + size)], shape))
            start += size
        gvp = [tf.reduce_sum(g * tangent) for (g, tangent) in zip(grads, tangents)]

        # Fisher-vector product
        fvp = flatgrad(gvp, params)

        ent = tf.reduce_mean(probtype.entropy(prob_np))
        kl = tf.reduce_mean(probtype.kl(oldprob_np, prob_np))

        losses = [surr, kl, ent]
        self.loss_names = ["surr", "kl", "ent"]

        args = [ob_no, act_na, adv_n, oldprob_np]

        self.compute_policy_gradient = TensorFlowLazyFunction(args, pg, session)
        self.compute_losses = TensorFlowLazyFunction(args, losses, session)
        self.compute_fisher_vector_product = TensorFlowLazyFunction([flat_tangent] + args, fvp, session)

    def __call__(self, paths):
        prob_np = concat([path["prob"] for path in paths])
        ob_no = concat([path["observation"] for path in paths])
        action_na = concat([path["action"] for path in paths])
        advantage_n = concat([path["advantage"] for path in paths])
        args = (ob_no, action_na, advantage_n, prob_np)

        cfg = self.cfg
        thprev = self.ezflat.get_params_flat()

        def fisher_vector_product(p):
            return self.compute_fisher_vector_product(p, *args) + cfg.TRPO.cg_damping * p

        g = self.compute_policy_gradient(*args)
        losses_before = self.compute_losses(*args)

        if np.allclose(g, 0):
            print("got zero gradient. not updating")
        else:
            stepdir = cg(fisher_vector_product, -g)
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / cfg.TRPO.max_kl)
            print("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            neggdotstepdir = -g.dot(stepdir)

            def loss(th):
                self.ezflat.set_params_flat(th)
                return self.compute_losses(*args)[0]

            success, theta = linesearch(loss, thprev, fullstep, neggdotstepdir/lm)
            print("success", success)
            self.ezflat.set_params_flat(theta)

        losses_after = self.compute_losses(*args)

        out = OrderedDict()
        for (lname, lbefore, lafter) in zipsame(self.loss_names, losses_before, losses_after):
            out[lname+"_before"] = lbefore
            out[lname+"_after"] = lafter
        return out


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
