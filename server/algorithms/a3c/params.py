class Params(object):
    def __init__(self, params):

        # action size for given game rom (18 fits ale boxing)
        self.action_size = params.get('action_size', 18)

        # local loop size for one episode
        self.episode_len = params.get('episode_len', 5)

        # to use GPU, set to the True
        self.use_GPU = params.get('gpu', False)

        # to use LSTM instead of FF, set to the True
        self.use_LSTM = params.get('lstm', False)

        # amount of maximum global steps to pass through the training
        self.max_global_step = params.get('max_global_step', 10 * 10 ** 7)

        self.INITIAL_LEARNING_RATE = params.get('initial_learning_rate', 7e-4)

        # discount factor for rewards
        self.GAMMA = params.get('rewards_gamma', 0.99)

        # entropy regularization constant
        self.ENTROPY_BETA = params.get('entropy_beta', 0.01)

        # decay parameter for RMSProp
        self.RMSP_ALPHA = params.get('RMSProp', {}).get('decay', 0.99)

        # epsilon parameter for RMSProp
        self.RMSP_EPSILON = params.get('RMSProp', {}).get('epsilon', 0.1)

        # gradient norm clipping
        self.GRAD_NORM_CLIP = params.get('RMSProp', {}).get('gradient_norm_clipping', 40)
