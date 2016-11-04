from algos import *

# ================================================================
# Proximal Policy Optimization: SGD && L-BFGS
# ================================================================


class PpoLbfgsUpdater(EzFlat, EzPickle):
    options = [
        ("kl_target", float, 1e-2, "Desired KL divergence between old and new policy"),
        ("maxiter", int, 25, "Maximum number of iterations"),
        ("reverse_kl", int, 0, "kl[new, old] instead of kl[old, new]"),
        ("do_split", int, 0, "Do train/test split on batches")
    ]

    def __init__(self, stochpol, usercfg, session):
        EzPickle.__init__(self, stochpol, usercfg)
        cfg = update_default_config(self.options, usercfg)
        print("PPOUpdater", cfg)

        self.stochpol = stochpol
        self.cfg = cfg
        self.kl_coeff = 1.0
        kl_cutoff = cfg["kl_target"] * 2.0

        probtype = stochpol.probtype
        params = stochpol.trainable_variables
        EzFlat.__init__(self, params, session)

        ob_no = stochpol.input
        act_na = probtype.sampled_variable()
        adv_n = tf.placeholder(dtype, name='adv_n')
        kl_coeff = tf.placeholder(dtype, name='kl_coeff')
        # kl_coeff = T.scalar("kl_coeff")

        # Probability distribution:
        prob_np = stochpol.get_output()
        oldprob_np = probtype.prob_variable()

        p_n = probtype.likelihood(act_na, prob_np)
        oldp_n = probtype.likelihood(act_na, oldprob_np)
        N = tf.shape(ob_no)[0]
        N_float = tf.cast(N, tf.float32)

        ent = tf.reduce_mean(probtype.entropy(prob_np))
        if cfg["reverse_kl"]:
            kl = tf.reduce_mean(probtype.kl(prob_np, oldprob_np))
        else:
            kl = tf.reduce_mean(probtype.kl(oldprob_np, prob_np))

        # Policy gradient:
        surr = -tf.reduce_mean(tf.mul((p_n / oldp_n), adv_n))
        # surr = (-1.0 / N) * (p_n / oldp_n).dot(adv_n)

        '''
        if tf.greater(kl, kl_cutoff):
            pensurr = surr + kl_coeff * kl + 1000 * tf.square(kl - kl_cutoff)
        else:
            pensurr = surr + kl_coeff * kl
        '''
        pensurr = tf.select(tf.greater(kl, kl_cutoff),
                            surr + kl_coeff * kl + 1000 * tf.square(kl - kl_cutoff),
                            surr + kl_coeff * kl)
        # pensurr = surr + kl_coeff * kl + 1000 * tf.greater(kl, kl_cutoff) * tf.square(kl - kl_cutoff)
        # pensurr = surr + kl_coeff * kl + 1000 * (kl > kl_cutoff) * tf.square(kl-kl_cutoff)
        g = flatgrad(pensurr, params)

        losses = [surr, kl, ent]
        self.loss_names = ["surr", "kl", "ent"]

        args = [ob_no, act_na, adv_n, oldprob_np]

        self.compute_lossgrad = TensorFlowTheanoFunction([kl_coeff] + args, [pensurr, g], session)
        self.compute_losses = TensorFlowTheanoFunction(args, losses, session)

    def __call__(self, paths):
        cfg = self.cfg
        prob_np = concat([path["prob"] for path in paths])
        ob_no = concat([path["observation"] for path in paths])
        action_na = concat([path["action"] for path in paths])
        advantage_n = concat([path["advantage"] for path in paths])

        N = tf.shape(ob_no)[0]
        N_float = tf.cast(N, tf.float32)

        train_stop = int(0.75 * N_float) if cfg["do_split"] else N_float
        train_sli = slice(0, train_stop)
        test_sli = slice(train_stop, None)

        train_args = (ob_no[train_sli], action_na[train_sli], advantage_n[train_sli], prob_np[train_sli])

        thprev = self.get_params_flat()

        def lossandgrad(th):
            self.set_params_flat(th)
            l, g = self.compute_lossgrad(self.kl_coeff, *train_args)
            g = g.astype('float64')
            return l, g

        train_losses_before = self.compute_losses(*train_args)
        if cfg["do_split"]:
            test_args = (ob_no[test_sli], action_na[test_sli], advantage_n[test_sli], prob_np[test_sli])
            test_losses_before = self.compute_losses(*test_args)

        theta, _, opt_info = scipy.optimize.fmin_l_bfgs_b(lossandgrad, thprev, maxiter=cfg["maxiter"])
        del opt_info['grad']
        print(opt_info)

        self.set_params_flat(theta)
        train_losses_after = self.compute_losses(*train_args)

        if cfg["do_split"]:
            test_losses_after = self.compute_losses(*test_args)
        klafter = train_losses_after[self.loss_names.index("kl")]

        if klafter > 1.3*self.cfg["kl_target"]:
            self.kl_coeff *= 1.5
            print("Got KL=%.3f (target %.3f). Increasing penalty coeff => %.3f."
                  % (klafter, self.cfg["kl_target"], self.kl_coeff))
        elif klafter < 0.7*self.cfg["kl_target"]:
            self.kl_coeff /= 1.5
            print("Got KL=%.3f (target %.3f). Decreasing penalty coeff => %.3f."
                  % (klafter, self.cfg["kl_target"], self.kl_coeff))
        else:
            print("KL=%.3f is close enough to target %.3f." % (klafter, self.cfg["kl_target"]))

        info = OrderedDict()
        for (name, lossbefore, lossafter) in zipsame(self.loss_names, train_losses_before, train_losses_after):
            info[name+"_before"] = lossbefore
            info[name+"_after"] = lossafter
            info[name+"_change"] = lossafter - lossbefore
        if cfg["do_split"]:
            for (name,lossbefore, lossafter) in zipsame(self.loss_names, test_losses_before, test_losses_after):
                info["test_"+name+"_before"] = lossbefore
                info["test_"+name+"_after"] = lossafter
                info["test_"+name+"_change"] = lossafter - lossbefore

        return info

class PpoSgdUpdater(EzPickle):
    options = [
        ("kl_target", float, 1e-2, "Desired KL divergence between old and new policy"),
        ("epochs", int, 10, ""),
        ("stepsize", float, 1e-3, ""),
        ("do_split", int, 0, "do train/test split"),
        ("kl_cutoff_coeff", float, 1000.0, "")
    ]

    def __init__(self, stochpol, usercfg, session):
        EzPickle.__init__(self, stochpol, usercfg)
        cfg = update_default_config(self.options, usercfg)
        print("PPOUpdater", cfg)

        self.stochpol = stochpol
        self.cfg = cfg
        self.kl_coeff = 1.0
        kl_cutoff = cfg["kl_target"] * 2.0

        probtype = stochpol.probtype
        params = stochpol.trainable_variables
        old_params = [v.get_value() for v in stochpol.trainable_variables]  # ?!
        # old_params = [theano.shared(v.get_value()) for v in stochpol.trainable_variables]

        ob_no = stochpol.input
        act_na = probtype.sampled_variable()
        adv_n = tf.placeholder(dtype, name='adv_n')
        kl_coeff = tf.placeholder(dtype, name='kl_coeff')
        #  = T.scalar("kl_coeff")

        # Probability distribution:
        self.loss_names = ["surr", "kl", "ent"]
        prob_np = stochpol.get_output()
        oldprob_np = tf.assign(stochpol.get_output(), replace=dict(zipsame(params, old_params)))
        # oldprob_np = theano.clone(stochpol.get_output(), replace=dict(zipsame(params, old_params)))

        p_n = probtype.likelihood(act_na, prob_np)
        oldp_n = probtype.likelihood(act_na, oldprob_np)
        # N = tf.shape(ob_no)[0]
        # N_float = tf.cast(N, tf.float32)

        ent = tf.reduce_mean(probtype.entropy(prob_np))
        kl = tf.reduce_mean(probtype.kl(oldprob_np, prob_np))

        # Policy gradient:
        surr = -tf.reduce_mean(tf.mul((p_n / oldp_n), adv_n))
        # surr = (-1.0 / N) * (p_n / oldp_n).dot(adv_n)
        train_losses = [surr, kl, ent]

        # training
        args = [ob_no, act_na, adv_n]
        surr, kl = train_losses[:2]
        pensurr = surr + kl_coeff * kl + cfg["kl_cutoff_coeff"] * (kl > kl_cutoff) * tf.square(kl - kl_cutoff)

        # troubles below
        self.train = theano.function([kl_coeff] + args, train_losses,
                                     updates=stochpol.get_updates()
                                     + adam_updates(pensurr, params, learning_rate=cfg.stepsize).items(),
                                     **FNOPTS)

        self.test = TensorFlowTheanoFunction(args, train_losses, session)  # XXX
        self.update_old_net = 0
        # self.update_old_net = theano.function([], [], updates=zip(old_params, params))
