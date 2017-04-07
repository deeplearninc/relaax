import grpc

import bridge_pb2
import bridge_message


class BridgeConnection(object):
    def __init__(self, server):
        self.__stub = bridge_pb2.BridgeStub(grpc.insecure_channel('%s:%d' % server))

    def __getattr__(self, name):
        return BridgeConnectionMethod(self.__stub, name)


class BridgeConnectionMethod(object):
    def __init__(self, stub, op):
        self.stub = stub
        self.op = op

    def __call__(self, **kwargs):
        feed_dict = kwargs
        messages = bridge_message.BridgeMessage.serialize([self.op, feed_dict])
        result = self.stub.Run(messages)
        return bridge_message.BridgeMessage.deserialize(result)
