from . import *
from relaax.common.python.config.loaded_config import options


class PGConfig(object):
    @classmethod
    def preprocess(self):
        config = options.get('algorithm')
        # flatten state
        config.state_size = np.prod(np.array(config.state_shape))
        return config
