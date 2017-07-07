from __future__ import absolute_import
from builtins import object
import concurrent
import grpc

from . import bridge_pb2
from . import bridge_message


class MetricsBridgeServer(object):
    def __init__(self, bind, metrics_server):
        self.server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=1))
        bridge_pb2.add_BridgeServicer_to_server(Servicer(metrics_server), self.server)
        self.server.add_insecure_port('%s:%d' % bind)

    def start(self):
        self.server.start()


class Servicer(bridge_pb2.BridgeServicer):
    def __init__(self, metrics_server):
        self.metrics_server = metrics_server

    def SetX(self, request, context):
        self.metrics_server.set_x(request.x)
        return bridge_pb2.NullMessage()

    def StoreMetric(self, request_iterator, context):
        data = bridge_message.BridgeMessage.deserialize(request_iterator)
        getattr(self.metrics_server.metrics, data['method'])(**data['kwargs'])
        return bridge_pb2.NullMessage()
