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

    def __init__(self, stochpol, usercfg):
        EzPickle.__init__(self, stochpol, usercfg)
        cfg = update_default_config(self.options, usercfg)
        print("PPOUpdater", cfg)


class PpoSgdUpdater(EzPickle):
    options = [
        ("kl_target", float, 1e-2, "Desired KL divergence between old and new policy"),
        ("epochs", int, 10, ""),
        ("stepsize", float, 1e-3, ""),
        ("do_split", int, 0, "do train/test split"),
        ("kl_cutoff_coeff", float, 1000.0, "")
    ]

    def __init__(self, stochpol, usercfg):
        EzPickle.__init__(self, stochpol, usercfg)
        cfg = update_default_config(self.options, usercfg)
        print("PPOUpdater", cfg)