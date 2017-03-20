import io
import sys
import json
import numpy
import struct
import base64

from twisted.python import log
from twisted.internet import defer
from twisted.internet import reactor
from twisted.internet.protocol import ClientFactory, Protocol
from twisted.protocols.basic import NetstringReceiver
from autobahn.twisted.websocket import WebSocketServerProtocol, WebSocketServerFactory

from config import options
from rlx_message import RLXMessage as rlxm

class ProxyClient(NetstringReceiver):
    
    def __init__(self,client_id,cli_queue,srv_queue):
        self.client_id = client_id
        self.cli_queue = cli_queue
        self.srv_queue = srv_queue

    def connectionMade(self):
        log.msg("Proxy Client connected to peer for client: ", self.client_id)
        self.cli_queue.get().addCallback(self.serverDataReceived)

    def stringReceived(self, data):
        msg = rlxm.client_from_wire(data,True)
        msg['id'] = self.client_id
        self.srv_queue.put(json.dumps(msg))
    
    def serverDataReceived(self,data):
        self.transport.write(data)
        self.cli_queue.get().addCallback(self.serverDataReceived)

    def connectionLost(self, reason):
        log.msg("Proxy Client connection lost for client: ", self.client_id)
        self.cli_queue = None

    def disconnect(self):
        log.msg("Proxy Client disconnecting client: ", self.client_id)
        self.transport.loseConnection()
        
class ProxyClientFactory(ClientFactory):

    def __init__(self,ws,client_id,srv_queue):
        self.ws = ws
        self.client_id = client_id
        self.cli_queue = defer.DeferredQueue()
        self.srv_queue = srv_queue

    def buildProtocol(self, addr):
        client = ProxyClient(
            self.client_id,
            self.cli_queue,
            self.srv_queue)
        self.ws.addClient(self.client_id,client)
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

    def addClient(self,client_id,client):
        self.clients[client_id] = client
        log.msg("Added proxy client for: ", client_id, self.clients)

    def removeClient(self,client_id):
        if client_id in self.clients:
            del self.clients[client_id]
            log.msg("Removed proxy client for: ", client_id, self.clients)

    def onConnect(self,request):
        log.msg("Client connecting: {0}".format(request.peer))
        self.srv_queue.get().addCallback(self.clientDataReceived)

    def connectRLX(self,client_id,srv_queue):
        factory = ProxyClientFactory(self,client_id,srv_queue)
        reactor.connectTCP(options.rlx_address[0], int(options.rlx_address[1]), factory)
        return factory.cli_queue

    def onMessage(self,payload,isBinary):
        msg = json.loads(payload)
        state = msg['state']
        client = self.clients.get(msg['id'])
        if client == None:
            verb = 'act'
            reward = None
            cli_queue = self.connectRLX(msg['id'],self.srv_queue)
        else:
            verb = 'reward_and_act'
            reward = msg['reward']
            cli_queue = client.cli_queue
        cli_queue.put(rlxm.client_to_wire(verb,state=state,reward=reward))

    def clientDataReceived(self,data):
        self.sendMessage(data)
        self.srv_queue.get().addCallback(self.clientDataReceived)

    def onClose(self, wasClean, code, reason):
        log.msg("WebSocket connection closed: {0}".format(reason))
        for client in self.clients.itervalues():
            client.disconnect()

if __name__ == '__main__':
    log.startLogging(sys.stdout)
    address = "ws://%s:%s" % tuple(options.ws_address)
    log.msg("Starting Web Socket server on %s" % address)   
    factory = WebSocketServerFactory(address)
    factory.protocol = WsServerProtocol
    # factory.setProtocolOptions(maxConnections=2)
    reactor.listenTCP(int(options.ws_address[1]), factory)
    log.msg("Expecting RLX server on %s:%s" % tuple(options.rlx_address))
    reactor.run()
