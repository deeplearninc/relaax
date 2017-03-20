import numpy as np

from twisted.internet import task
from twisted.internet.defer import Deferred
from twisted.internet.protocol import ClientFactory
from twisted.protocols.basic import NetstringReceiver

from relaax.common.rlx_message import RLXMessage

class SampleClient(NetstringReceiver):

    def __init__(self):
        self.updates = 0

    def init(self):
        self._send({'command':'init'})

    def update(self,reward=None,state=None,terminal=False):
        self._send({'command':'update','reward':reward,'state':state,'terminal':terminal})

    def reset(self): 
        self._send({'command':'reset'})

    def _send(self,data):
        self.sendString(RLXMessage.to_wire(data))

    def _receive(self,data):
        return RLXMessage.from_wire(data)

    def connectionMade(self):
        ### initialize agent
        self.init()

    def stringReceived(self, data):
        msg = self._receive(data)
        print("received from agent:", msg)

        if msg['response'] == 'action':
            # perform action on environment  
            print "action: ", msg['data']

        if self.updates < 1:
            self.updates += 1
            ### update agent
            self.update(reward=-1.1,state=[3.3,4.4,5.5],terminal=False)

        elif msg['response'] == 'done' or msg['response'] == 'error':
            self.transport.loseConnection()

        else:
            # reset agent (should return 'done' and we'll close connection)
            self.reset()

class SampleClientFactory(ClientFactory):
    protocol = SampleClient

    def __init__(self):
        self.done = Deferred()

    def clientConnectionFailed(self, connector, reason):
        print('connection failed:', reason.getErrorMessage())
        self.done.errback(reason)

    def clientConnectionLost(self, connector, reason):
        print('connection lost:', reason.getErrorMessage())
        self.done.callback(None)

def main(reactor):
    factory = SampleClientFactory()
    reactor.connectTCP('localhost', 7001, factory)
    return factory.done

if __name__ == '__main__':
    task.react(main)
