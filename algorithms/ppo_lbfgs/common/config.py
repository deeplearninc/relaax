import relaax.algorithm_base.config_base


class Config(relaax.algorithm_base.config_base.ConfigBase):
    def __init__(self, config):

        # action size for the given environment (4 fits to bipedal_walker)
        self.action_size = config.get('action_size', 4)

        # size of the input observation
        self.state_size = config.get('state_size', [24])

        # number of hidden units for each hidden layer of MLP (sequentially)
        self.hidden_layers_sizes = config.get('hidden_layers_sizes', [64, 64])

        # activation function, which is used for each layer of MLP
        self.activation = config.get('activation', 'tanh')

        # switch between continuous and discrete action spaces (Box is continuous)
        self.discrete = config.get('discrete', False)

        # whether to do a running average filter of the incoming observations and rewards
        self.use_filter = config.get('use_filter', True)

        # set to true to collect experience without blocking the updater
        self.async_collect = config.get('async', False)

        # POLICY GRADIENT OPTIONS
        # maximum length of trajectories (length in steps for one round in environment)
        self.timestep_limit = config.get('PG_OPTIONS', {}).get('timestep_limit', 2000)

        # number of updates for collected batch (training length)
        self.n_iter = config.get('PG_OPTIONS', {}).get('n_iter', 10**4)

        # number of experience to collect before update (batch size)
        self.timesteps_per_batch = config.get('PG_OPTIONS', {}).get('timesteps_per_batch', 2*10**4)

        # discount factor for rewards
        self.GAMMA = config.get('PG_OPTIONS', {}).get('rewards_gamma', 0.995)

        # lambda parameter from generalized advantage estimation
        self.LAMBDA = config.get('PG_OPTIONS', {}).get('gae_lambda', 0.97)

        # PPO OPTIONS
        # Desired KL divergence between old and new policy
        self.kl_target = config.get('PPO', {}).get('kl_target', 1e-2)

        # KL[new, old] instead of KL[old, new]
        self.reverse_kl = config.get('PPO', {}).get('reverse_kl', 0)

        # Maximum number of iterations
        self.maxiter = config.get('PPO', {}).get('maxiter', 25)

        # Do train/test split on batches
        self.do_split = config.get('PPO', {}).get('do_split', 0)
