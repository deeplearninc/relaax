import tensorflow as tf

import relaax.algorithm_base.parameter_server

from . import network


class ParameterServer(relaax.algorithm_base.parameter_server.ParameterServer):
    def __init__(self, config, saver):
        super(ParameterServer, self).__init__(config, network.make(config), saver)
