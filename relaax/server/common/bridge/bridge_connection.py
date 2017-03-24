import grpc

import bridge_pb2

from bridge_serializer import BridgeSerializer


class BridgeConnection(object):
    def __init__(self, server_url):
        self._stub = bridge_pb2.BridgeStub(grpc.insecure_channel(server_url))

    def run(self, ops, feed_dict):
        messages = BridgeSerializer.serialize([ops, feed_dict])
        result = self._stub.Run(messages)
        return list(BridgeSerializer.deserialize(result))
