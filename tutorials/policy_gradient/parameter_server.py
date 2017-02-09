import tensorflow as tf

import relaax.algorithm_base.bridge_base
import relaax.algorithm_base.parameter_server_base


class ParameterServer(relaax.algorithm_base.parameter_server_base.ParameterServerBase):
    def __init__(self, config, saver, metrics):
        pass

class _Bridge(relaax.algorithm_base.bridge_base.BridgeBase):
    def __init__(self, config, metrics, network, session):
        self._config = config
