import relaax.algorithm_base.da3c.parameter_server

from . import network


class ParameterServer(relaax.algorithm_base.da3c.parameter_server.ParameterServer):
    def __init__(self, config, saver, metrics):
        super(ParameterServer, self).__init__(config, network.make(config), saver, metrics)
