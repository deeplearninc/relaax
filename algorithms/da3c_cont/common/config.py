import relaax.algorithm_base.config_base


class Config(relaax.algorithm_base.config_base.ConfigBase):
    def __init__(self, config):

        # action size for the given environment (4 fits to bipedal_walker)
        self.action_size = config.get('action_size', 4)

        # size of the input observation
        self.state_size = config.get('state_size', [24])

        # number of consecutive observations to stack in state
        self.history_len = config.get('history_len', 1)

        # local loop size for one episode
        self.episode_len = config.get('episode_len', 5)

        # to use GPU, set to the True
        self.use_GPU = config.get('gpu', False)

        # to use LSTM instead of FF, set to the True
        self.use_LSTM = config.get('lstm', True)

        # amount of maximum global steps to pass through the training
        self.max_global_step = config.get('max_global_step', 8 * 10 ** 7)

        # useful for non-episodic (non-terminal) environments, it prints out the accumalated
        # reward each defined interval, for episodic environments it is recommended to set
        # this to 0 to print out accumulated reward at each episode end
        self.reward_interval = config.get('reward_count_interval', 0)

        # initial learning rate from which we start to anneal
        self.INITIAL_LEARNING_RATE = config.get('initial_learning_rate', 5e-5)

        # discount factor for rewards
        self.GAMMA = config.get('rewards_gamma', 0.99)

        # entropy regularization constant
        self.ENTROPY_BETA = config.get('entropy_beta', 1e-3)

        # decay parameter for RMSProp
        self.RMSP_ALPHA = config.get('RMSProp', {}).get('decay', 0.99)

        # epsilon parameter for RMSProp
        self.RMSP_EPSILON = config.get('RMSProp', {}).get('epsilon', 0.1)

        # gradient norm clipping
        self.GRAD_NORM_CLIP = config.get('RMSProp', {}).get('gradient_norm_clipping', 40)
