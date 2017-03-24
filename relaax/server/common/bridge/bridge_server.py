import concurrent
import grpc

import bridge_pb2

from bridge_serializer import BridgeSerializer


class BridgeServer(object):
    def __init__(self, bind, session):
        self.server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=1))
        bridge_pb2.add_BridgeServicer_to_server(Servicer(session), self.server)
        self.server.add_insecure_port('%s:%d' % bind)

    def start(self):
        self.server.start()


class Servicer(bridge_pb2.BridgeServicer):
    def __init__(self, session):
        self.session = session

    def Run(self, request_iterator, context):
        ops, feed_dict = BridgeSerializer.deserialize(request_iterator)
        result = self.session.run(ops, feed_dict)
        return BridgeSerializer.serialize(result)
