import grpc

import bridge_pb2

from bridge_protocol import BridgeProtocol


class BridgeConnection(object):
    def __init__(self, server_url):
        self._stub = bridge_pb2.BridgeStub(grpc.insecure_channel(server_url))

    def run(self, ops, feed_dict={}):
        messages = BridgeProtocol.build_arg_messages(ops, feed_dict)
        result = self._stub.Run(messages)
        return list(BridgeProtocol.parse_result_messages(result))
