import relaax.algorithm_base.config_base


class Config(relaax.algorithm_base.config_base.ConfigBase):

    def __init__(self, config):
        # action size for the given environment (gym's Pong)
        self.action_size = config.get('action_size', 3)

        # pass flattened or n-element list for an image-like state
        self.state_size = config.get('state_size', [84, 84])

        # if true, input == difference from previous state
        self.preprocess = config.get('preprocess', False)

        # # elements == hidden layers, each number == neurons
        self.layers_size = config.get('hidden_layers_size', [200, 200])

        # how many steps perform before a param update
        self.batch_size = config.get('batch_size', 10)

        # maximum global step to stop the training when it is reached
        self.max_global_step = config.get('max_global_step', 10 * 10 ** 7)

        # learning rate which we use through whole training
        self.learning_rate = config.get('learning_rate', 1e-4)

        # discount factor for rewards
        self.GAMMA = config.get('rewards_gamma', 0.99)

        # entropy regularization constant
        self.ENTROPY_BETA = config.get('entropy_beta', 0.01)

        # decay parameter for RMSProp
        self.RMSP_DECAY = config.get('RMSProp', {}).get('decay', 0.99)

        # epsilon parameter for RMSProp
        self.RMSP_EPSILON = config.get('RMSProp', {}).get('epsilon', 1e-5)
