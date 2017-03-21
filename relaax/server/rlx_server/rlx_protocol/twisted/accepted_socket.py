import logging

from twisted.internet import abstract
from twisted.internet.tcp import Server
from twisted.internet import address, fdesc

log = logging.getLogger(__name__)

# AcceptedSocket


class AcceptedSocket():
    transport = Server

    def __init__(self, skt, addr, factory, reactor, interface=''):
        self.skt = skt
        self.addr = addr
        self.factory = factory
        self.reactor = reactor
        self._realPortNumber = skt.getsockname()[1]
        self._addressType = address.IPv4Address
        if abstract.isIPv6Address(interface):
            self._addressType = address.IPv6Address

    def _buildAddr(self, addr):
        host, port = addr[:2]
        return self._addressType('TCP', host, port)

    def start(self):
        fdesc._setCloseOnExec(self.skt.fileno())
        protocol = self.factory.buildProtocol(self._buildAddr(self.addr))
        if protocol is None:
            log.error("Can't create protocol")
            return False
        transport = self.transport(
            self.skt, protocol, self.addr, self, 1, self.reactor)
        protocol.makeConnection(transport)
        return True
