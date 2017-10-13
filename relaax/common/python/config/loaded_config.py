from builtins import object
import sys


class AlgoritmConfig(object):

    @classmethod
    def select_config(cls):
        rlx = 'relaax.server.rlx_server.rlx_config'
        ps = 'relaax.server.parameter_server.parameter_server_config'

        if rlx in sys.modules:
            from relaax.server.rlx_server.rlx_config import options
            return options
        elif ps in sys.modules:
            from relaax.server.parameter_server.parameter_server_config import options
            return options
        else:
            from relaax.common.algorithms.python.dev_config import options
            return options


options = AlgoritmConfig.select_config()
