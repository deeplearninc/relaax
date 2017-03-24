import grpc

import bridge_pb2

from bridge_message import BridgeMessage


class BridgeConnection(object):
    def __init__(self, server):
        self._stub = bridge_pb2.BridgeStub(grpc.insecure_channel('%s:%d' % server))

    def run(self, ops, feed_dict):
        messages = BridgeMessage.serialize([ops, feed_dict])
        result = self._stub.Run(messages)
        return list(BridgeMessage.deserialize(result))
