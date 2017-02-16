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
        self.use_filters = config.get('use_filters', True)

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

        # TRPO OPTIONS
        # Add multiple of the identity to Fisher matrix during CG
        self.cg_damping = config.get('TRPO', {}).get('cg_damping', 0.1)

        # KL divergence between old and new policy (averaged over state-space)
        self.max_kl = config.get('TRPO', {}).get('gradient_norm_clipping', 0.01)