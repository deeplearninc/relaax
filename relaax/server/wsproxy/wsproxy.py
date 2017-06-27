from __future__ import absolute_import
import json
import numpy
import logging

from twisted.internet import defer
from twisted.internet import reactor
from twisted.internet.protocol import ClientFactory
from twisted.protocols.basic import NetstringReceiver
from autobahn.twisted.websocket import WebSocketServerProtocol
from autobahn.twisted.websocket import WebSocketServerFactory

from relaax.common.rlx_message import RLXMessage
from .wsproxy_config import options

log = logging.getLogger(__name__)


class ProxyClient(NetstringReceiver):

    def __init__(self, client_id, cli_queue, srv_queue):
        self.client_id = client_id
        self.cli_queue = cli_queue
        self.srv_queue = srv_queue

    def connectionMade(self):
        log.debug("Proxy Client connected to peer for client: {0}".format(self.client_id))
        self.cli_queue.get().addCallback(self.serverDataReceived)

    def stringReceived(self, data):
        msg = RLXMessage.from_wire(data)
        msg['sid'] = self.client_id
        if 'data' in msg and isinstance(msg['data'], numpy.ndarray):
            msg['data'] = msg['data'].tolist()
        self.srv_queue.put(json.dumps(msg))

    def serverDataReceived(self, data):
        if data['command'] == 'disconnect':
            self.disconnect()
        else:
            self.sendString(RLXMessage.to_wire(data).encode())
            self.cli_queue.get().addCallback(self.serverDataReceived)

    def connectionLost(self, reason):
        log.debug("Proxy Client connection lost for client: {0}".format(self.client_id))
        self.cli_queue = None

    def disconnect(self):
        log.debug("Proxy Client disconnecting client: {0}".format(self.client_id))
        self.transport.loseConnection()


class ProxyClientFactory(ClientFactory):

    def __init__(self, ws, client_id, srv_queue):
        self.ws = ws
        self.client_id = client_id
        self.cli_queue = defer.DeferredQueue()
        self.srv_queue = srv_queue

    def buildProtocol(self, addr):
        client = ProxyClient(
            self.client_id,
            self.cli_queue,
            self.srv_queue)
        self.ws.addClient(self.client_id, client)
        return client

    def clientConnectionLost(self, connector, reason):
        self.ws.removeClient(self.client_id)

    def clientConnectionFailed(self, connector, reason):
        self.ws.removeClient(self.client_id)


class WsServerProtocol(WebSocketServerProtocol):

    def __init__(self):
        super(WebSocketServerProtocol, self).__init__()
        self.clients = {}
        self.srv_queue = defer.DeferredQueue()

    def addClient(self, client_id, client):
        self.clients[client_id] = client
        log.debug("Added proxy client. Client sid: {0}".format(client_id))

    def removeClient(self, client_id):
        if client_id in self.clients:
            del self.clients[client_id]
            log.debug("Removed proxy client for: {0}, {1}".format(client_id, self.clients))

    def onConnect(self, request):
        log.debug("Client connecting: {0}".format(request.peer))
        self.srv_queue.get().addCallback(self.clientDataReceived)

    def onClose(self, wasClean, code, reason):
        if wasClean:
            log.debug("WebSocket connection closed...")
        else:
            log.debug("WebSocket connection was broken. Closing code [{0}], reason: {1}".format(code, reason))
        for client in self.clients.values():
            client.disconnect()

    def connectRLX(self, client_id, srv_queue):
        factory = ProxyClientFactory(self, client_id, srv_queue)
        reactor.connectTCP(options.rlx_server[0], options.rlx_server[1], factory)
        return factory.cli_queue

    def onMessage(self, payload, isBinary):
        msg = json.loads(payload.decode("utf-8"))
        client = self.clients.get(msg['sid'])
        if client is None:
            cli_queue = self.connectRLX(msg['sid'], self.srv_queue)
        else:
            cli_queue = client.cli_queue
        cli_queue.put(msg)

    def clientDataReceived(self, data):
        self.sendMessage(data.encode())
        self.srv_queue.get().addCallback(self.clientDataReceived)


def main():
    address = "ws://%s:%s" % options.bind
    log.info("Starting Web Socket server on %s" % address)
    factory = WebSocketServerFactory(address)
    factory.protocol = WsServerProtocol
    # factory.setProtocolOptions(maxConnections=2)
    reactor.listenTCP(int(options.bind[1]), factory)
    log.info("Expecting RLX server on %s:%s" % options.rlx_server)
    reactor.run()

if __name__ == '__main__':
    main()
