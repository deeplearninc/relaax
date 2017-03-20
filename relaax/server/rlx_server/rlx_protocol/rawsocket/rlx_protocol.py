import traceback

import logging
log = logging.getLogger(__name__)

from relaax.common.rlx_message import RLXMessage as rlxm
from relaax.server.rlx_server.rlx_agent_proxy import RLXAgentProxy

from relaax.common.rlx_netstring import NetString, NetStringClosed

class RLXProtocol(NetString):

    def __init__(self, skt, address):
        NetString.__init__(self,skt)
        self.agent = RLXAgentProxy()
        self.address = address

    def connectionMade(self):
        pass

    def stringReceived(self, data):
        msg = rlxm.from_wire(data)
        res = self.agent.dataReceived(msg)
        self.sendString(res)

    def sendString(self, data):
        self.writeString(rlxm.to_wire(data))

    def connectionLost(self, reason):
        pass

    def protocolLoop(self):
        reason = None
        try:
            self.connectionMade()
            while True:
                data = self.readString()
                self.stringReceived(data)                  

        except NetStringClosed as e:
            reason = "Connection dropped"
            log.debug("Raw Socket Connection dropped on connection %s:%d"%self.address)
        except Exception as e:
            reason = "Unknown error"
            log.error("Something crashed in the protocol for connection %s:%d"%self.address)
            log.error(traceback.format_exc())
        finally:
            self.connectionLost(reason)

def adoptConnection(skt,addr):
    RLXProtocol(skt,addr).protocolLoop()


