class Params(object):
    def __init__(self):
        self.default_params = {'game_rom': None,
                               'action_size': None,
                               'threads_cnt': 8,
                               'episode_len': 5,
                               'use_GPU': False,
                               'use_LSTM': True,
                               'max_global_step': 10 * 10 ** 7}
        self.game_rom = None     # name of the given game rom
        self.action_size = None  # action size for given game rom
        self.threads_cnt = None  # number of parallel training agents
        self.episode_len = None  # local loop size for one episode
        self.use_GPU = None      # to use GPU, set to the True
        self.use_LSTM = None     # to use LSTM instead of FF, set to the True
        self.max_global_step = None  # amount of maximum global steps to pass through the training

        self.RMSP_ALPHA = 0.99          # decay parameter for RMSProp
        self.RMSP_EPSILON = 0.1         # epsilon parameter for RMSProp
        self.INITIAL_ALPHA_LOW = 1e-4   # log_uniform low limit for learning rate
        self.INITIAL_ALPHA_HIGH = 1e-2  # log_uniform high limit for learning rate
        self.INITIAL_ALPHA_LOG_RATE = 0.4226  # log_uniform interpolate rate for learning rate (around 7 * 10^-4)
        self.GAMMA = 0.99               # discount factor for rewards
        self.ENTROPY_BETA = 0.01        # entropy regularization constant
        self.GRAD_NORM_CLIP = 40.0      # gradient norm clipping
