import numpy as np

from relaax.common.python.config.loaded_config import options


class PGConfig(object):
    @classmethod
    def preprocess(cls):
        return options.get('algorithm')


config = PGConfig.preprocess()
