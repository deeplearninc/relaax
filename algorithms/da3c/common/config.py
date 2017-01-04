class Config(object):
    def __init__(self, config):

        # action size for given game rom (18 fits ale boxing)
        self.action_size = config.get('action_size', 18)

        # local loop size for one episode
        self.episode_len = config.get('episode_len', 5)

        # number of consecutive observations to stack in state
        self.history_len = config.get('history_len', 4)

        # to use GPU, set to the True
        self.use_GPU = config.get('gpu', False)

        # to use LSTM instead of FF, set to the True
        self.use_LSTM = config.get('lstm', False)

        # amount of maximum global steps to pass through the training
        self.max_global_step = config.get('max_global_step', 10 * 10 ** 7)

        # learning rate from which we start to annealing
        self.INITIAL_LEARNING_RATE = config.get('initial_learning_rate', 7e-4)

        # discount factor for rewards
        self.GAMMA = config.get('rewards_gamma', 0.99)

        # entropy regularization constant
        self.ENTROPY_BETA = config.get('entropy_beta', 0.01)

        # decay parameter for RMSProp
        self.RMSP_ALPHA = config.get('RMSProp', {}).get('decay', 0.99)

        # epsilon parameter for RMSProp
        self.RMSP_EPSILON = config.get('RMSProp', {}).get('epsilon', 0.1)

        # gradient norm clipping
        self.GRAD_NORM_CLIP = config.get('RMSProp', {}).get('gradient_norm_clipping', 40)
