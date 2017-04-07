from relaax.server.parameter_server import parameter_server_base
from relaax.server.common import session

from pg_model import SharedParameters


class PGParameterServer(parameter_server_base.ParameterServerBase):
    def __init__(self):
        self.session = session.Session(SharedParameters())
        self.session.op_initialize()
