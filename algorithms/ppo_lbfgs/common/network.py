from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Lambda

from relaax.algorithm_lib.core import *


def make_mlps(config, session):
    K.set_session(session)

    policy_net = Sequential()
    for (i, layeroutsize) in enumerate(config.hidden_layers_sizes):
        input_shape = dict(input_shape=config.state_size) if i == 0 else {}
        policy_net.add(Dense(layeroutsize, activation=config.activation, **input_shape))
    if not config.discrete:
        policy_net.add(Dense(config.action_size))
        policy_net.add(Lambda(lambda x: x * 0.1))
        policy_net.add(ConcatFixedStd())
    else:
        policy_net.add(Dense(config.action_size, activation="softmax"))
        policy_net.add(Lambda(lambda x: x * 0.1))

    value_net = Sequential()
    for (i, layeroutsize) in enumerate(config.hidden_layers_sizes):
        # add one extra feature for timestep
        input_shape = dict(input_shape=(config.state_size[0]+1,)) if i == 0 else {}
        value_net.add(Dense(layeroutsize, activation=config.activation, **input_shape))
    value_net.add(Dense(1))

    return policy_net, value_net


def make_filters(config):
    if config.use_filters:
        obfilter = ZFilter(tuple(config.state_size), clip=5)
        rewfilter = ZFilter((), demean=False, clip=10)
    else:
        obfilter = IDENTITY
        rewfilter = IDENTITY
    return obfilter, rewfilter


def make_wrappers(config, policy_net, value_net, session):

    if not config.discrete:
        probtype = DiagGauss(config.action_size)
    else:
        probtype = Categorical(config.action_size)

    policy = StochPolicyKeras(policy_net, probtype, session)
    baseline = NnVf(value_net, config.timestep_limit, dict(mixfrac=0.1), session)

    return policy, baseline


class PpoLbfgsUpdater(EzFlat, EzPickle):
    def __init__(self, config, stochpol, session):
        EzPickle.__init__(self, stochpol, config)
        self.cfg = config
        self.stochpol = stochpol

        self.kl_coeff = 1.0
        kl_cutoff = config.kl_target * 2.0

        probtype = stochpol.probtype
        params = stochpol.trainable_variables
        EzFlat.__init__(self, params, session)

        ob_no = stochpol.input
        act_na = probtype.sampled_variable()
        adv_n = tf.placeholder(dtype, name='adv_n')
        kl_coeff = tf.placeholder(dtype, name='kl_coeff')

        # Probability distribution:
        prob_np = stochpol.get_output()
        oldprob_np = probtype.prob_variable()

        p_n = probtype.likelihood(act_na, prob_np)
        oldp_n = probtype.likelihood(act_na, oldprob_np)

        ent = tf.reduce_mean(probtype.entropy(prob_np))
        if config.reverse_kl:
            kl = tf.reduce_mean(probtype.kl(prob_np, oldprob_np))
        else:
            kl = tf.reduce_mean(probtype.kl(oldprob_np, prob_np))

        # Policy gradient:
        surr = -tf.reduce_mean(tf.mul((p_n / oldp_n), adv_n))   # tf.matmul --> .dot

        pensurr = tf.select(tf.greater(kl, kl_cutoff),
                            surr + kl_coeff * kl + 1000 * tf.square(kl - kl_cutoff),
                            surr + kl_coeff * kl)
        g = flatgrad(pensurr, params)

        losses = [surr, kl, ent]
        self.loss_names = ["surr", "kl", "ent"]

        args = [ob_no, act_na, adv_n, oldprob_np]

        self.compute_lossgrad = TensorFlowLazyFunction([kl_coeff] + args, [pensurr, g], session)
        self.compute_losses = TensorFlowLazyFunction(args, losses, session)

    def __call__(self, paths):
        prob_np = concat([path["prob"] for path in paths])
        ob_no = concat([path["observation"] for path in paths])
        action_na = concat([path["action"] for path in paths])
        advantage_n = concat([path["advantage"] for path in paths])

        train_stop = int(0.75 * len(ob_no)) if self.cfg.do_split else len(ob_no)
        train_sli = slice(0, train_stop)
        test_sli = slice(train_stop, None)

        train_args = (ob_no[train_sli], action_na[train_sli], advantage_n[train_sli], prob_np[train_sli])   # Fixed

        thprev = self.get_params_flat()

        def lossandgrad(th):
            self.set_params_flat(th)
            l, g = self.compute_lossgrad(self.kl_coeff, *train_args)
            g = g.astype('float64')
            return l, g

        train_losses_before = self.compute_losses(*train_args)
        if self.cfg.do_split:
            test_args = (ob_no[test_sli], action_na[test_sli], advantage_n[test_sli], prob_np[test_sli])
            test_losses_before = self.compute_losses(*test_args)

        theta, _, opt_info = scipy.optimize.fmin_l_bfgs_b(lossandgrad, thprev, maxiter=self.cfg.maxiter)
        del opt_info['grad']
        print(opt_info)

        self.set_params_flat(theta)
        train_losses_after = self.compute_losses(*train_args)

        if self.cfg.do_split:
            test_losses_after = self.compute_losses(*test_args)
        klafter = train_losses_after[self.loss_names.index("kl")]

        if klafter > 1.3*self.cfg.kl_target:
            self.kl_coeff *= 1.5
            print("Got KL=%.3f (target %.3f). Increasing penalty coeff => %.3f."
                  % (klafter, self.cfg.kl_target, self.kl_coeff))
        elif klafter < 0.7*self.cfg.kl_target:
            self.kl_coeff /= 1.5
            print("Got KL=%.3f (target %.3f). Decreasing penalty coeff => %.3f."
                  % (klafter, self.cfg.kl_target, self.kl_coeff))
        else:
            print("KL=%.3f is close enough to target %.3f." % (klafter, self.cfg.kl_target))

        info = OrderedDict()
        for (name, lossbefore, lossafter) in zipsame(self.loss_names, train_losses_before, train_losses_after):
            info[name+"_before"] = lossbefore
            info[name+"_after"] = lossafter
            info[name+"_change"] = lossafter - lossbefore
        if self.cfg.do_split:
            for (name,lossbefore, lossafter) in zipsame(self.loss_names, test_losses_before, test_losses_after):
                info["test_"+name+"_before"] = lossbefore
                info["test_"+name+"_after"] = lossafter
                info["test_"+name+"_change"] = lossafter - lossbefore

        return info
