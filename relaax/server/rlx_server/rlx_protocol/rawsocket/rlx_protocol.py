import logging
import traceback

from relaax.common.rlx_message import RLXMessage as rlxm
from relaax.server.rlx_server.rlx_agent_proxy import RLXAgentProxy
from relaax.common.rlx_netstring import NetString, NetStringClosed

log = logging.getLogger(__name__)


class RLXProtocol(NetString):

    def __init__(self, skt, address):
        NetString.__init__(self, skt)
        self.agent = RLXAgentProxy()
        self.address = address

    def connection_made(self):
        pass

    def string_received(self, data):
        msg = rlxm.from_wire(data)
        res = self.agent.data_received(msg)
        self.send_string(res)

    def send_string(self, data):
        self.write_string(rlxm.to_wire(data))

    def connection_lost(self, reason):
        pass

    def protocol_loop(self):
        reason = None
        try:
            self.connection_made()
            while True:
                data = self.read_string()
                self.string_received(data)

        except NetStringClosed:
            reason = 'Connection dropped'
            log.debug(('Raw Socket Connection dropped '
                       'on connection %s:%d') % self.address)
        except Exception:
            reason = "Unknown error"
            log.error(('Error in the protocol '
                       'for connection %s:%d') % self.address)
            log.error(traceback.format_exc())
        finally:
            self.connection_lost(reason)


def adoptConnection(skt, addr):
    RLXProtocol(skt, addr).protocol_loop()
