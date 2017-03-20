from relaax.common.python.config.base_config import BaseConfig

class DevConfig(BaseConfig):

    def __init__(self):
        super(DevConfig,self).__init__()

options = DevConfig().load()
