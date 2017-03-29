from relaax.server.parameter_server.parameter_server_base import ParameterServerBase

from .pgk_network import parameter_server_model


class PGParameterServer(ParameterServerBase):
    def __init__(self):
        # Build TF graph
        self.model = parameter_server_model()
