from builtins import object
from relaax.common.python.config.loaded_config import options


class PGConfig(object):
    @classmethod
    def preprocess(cls):
        return options.get('algorithm')


config = PGConfig.preprocess()
config.combine_gradient = options.get('algorithm/combine_gradient', 'fifo')
