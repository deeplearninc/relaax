from relaax.server.parameter_server import parameter_server_base
from relaax.server.common import session

import da3c_model


class DA3CParameterServer(parameter_server_base.ParameterServerBase):
    def __init__(self):
        self.session = session.Session(da3c_model.SharedParameters())
        self.session.op_initialize()
