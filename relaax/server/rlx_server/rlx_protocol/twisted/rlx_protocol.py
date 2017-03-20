import traceback
import logging
log = logging.getLogger(__name__)

from twisted.internet.defer import Deferred
from twisted.protocols.basic import NetstringReceiver
from twisted.internet.protocol import Factory, Protocol

from accepted_socket import AcceptedSocket
from relaax.common.rlx_message import RLXMessage as rlxm
from relaax.server.rlx_server.rlx_agent_proxy import RLXAgentProxy

### Protocol Implementation

class RLXProtocol(NetstringReceiver):

    def __init__(self,factory):
        self.factory = factory
        self.agent = RLXAgentProxy()

    def stringReceived(self, data):
        msg = rlxm.from_wire(data)
        res = self.agent.dataReceived(msg)
        self.sendString(rlxm.to_wire(res))

    def connectionLost(self, reason):
        self.factory.done.callback(None)


class RLXProtocolFactory(Factory):
    protocol = RLXProtocol

    def __init__(self):
        self.done = Deferred()

    def buildProtocol(self, addr):
        return RLXProtocol(self)

    @staticmethod
    def buildConnection(reactor,socket,address):
        factory = RLXProtocolFactory()
        adopted = AcceptedSocket(socket,address,factory,reactor)
        if not adopted.start():
            log.error("Failed to to build connection")
            factory.done.callback(None)
        return factory.done

def adoptConnection(socket,address):
    from twisted.internet import task    
    task.react(RLXProtocolFactory.buildConnection,(socket,address))

