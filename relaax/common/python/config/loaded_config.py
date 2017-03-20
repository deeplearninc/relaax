import sys

class AlgoritmConfig(object):

    @classmethod
    def select_config(cls):
        if 'relaax.server.rlx_server.rlx_config' in sys.modules:
            from relaax.server.rlx_server.rlx_config import options
            return options
        elif 'relaax.server.parameter_server.parameter_server_config' in sys.modules:
            from relaax.server.parameter_server.parameter_server_config import options
            return options
        else:
            from relaax.common.algorithms.python.dev_config import options
            return options

options = AlgoritmConfig.select_config()