from __future__ import print_function

import concurrent
import grpc
import numpy

import relaax.algorithm_base.parameter_server_base
import relaax.algorithm_base.bridge_base

from . import bridge_pb2


class Bridge(relaax.algorithm_base.bridge_base.BridgeBase):
    def parameter_server_stub(self, parameter_server_url):
        return _Stub(parameter_server_url)

    def start_parameter_server(self, address, service):
        server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=1))
        bridge_pb2.add_ParameterServerServicer_to_server(_Servicer(service), server)
        server.add_insecure_port(address)
        server.start()
        return server


class _Stub(relaax.algorithm_base.parameter_server_base.ParameterServerBase):
    def __init__(self, parameter_server):
        self._stub = bridge_pb2.ParameterServerStub(grpc.insecure_channel(parameter_server))
        self._metrics = _Metrics(self._stub)

    def increment_global_t(self):
        return self._stub.IncrementGlobalT(bridge_pb2.NullMessage()).n

    def apply_gradients(self, gradients):
        self._stub.ApplyGradients(_build_ndarrays_message(gradients))

    def get_values(self):
        return _parse_ndarrays_message(self._stub.GetValues(bridge_pb2.NullMessage()))

    def metrics(self):
        return self._metrics


class _Metrics(relaax.common.metrics.Metrics):
    def __init__(self, stub):
        self._stub = stub

    def scalar(self, name, y, x=None):
        if x is None:
            sm = bridge_pb2.ScalarMetric(
                name=name,
                y=y
            )
        else:
            sm = bridge_pb2.ScalarMetric(
                name=name,
                y=y,
                x=bridge_pb2.ScalarMetric.Arg(value=x)
            )
        self._stub.StoreScalarMetric(sm)


class _Servicer(bridge_pb2.ParameterServerServicer):
    def __init__(self, service):
        self._service = service

    def IncrementGlobalT(self, request, context):
        return bridge_pb2.Step(n=long(self._service.increment_global_t()))

    def ApplyGradients(self, request, context):
        self._service.apply_gradients(_parse_ndarrays_message(request))
        return bridge_pb2.NullMessage()

    def GetValues(self, request, context):
        return _build_ndarrays_message(self._service.get_values())

    def StoreScalarMetric(self, request, context):
        x = None
        if request.HasField('x'):
            x = request.x.value
        self._service.metrics().scalar(name=request.name, y=request.y, x=x)
        return bridge_pb2.NullMessage()


def _build_ndarrays_message(arrays):
    return bridge_pb2.NdArrays(arrays=[
        bridge_pb2.NdArrays.NdArray(
            dtype=str(a.dtype),
            shape=a.shape,
            data=a.tobytes()
        )
        for a in arrays
    ])


def _parse_ndarrays_message(message):
    return [
        numpy.ndarray(
            shape=a.shape,
            dtype=numpy.dtype(a.dtype),
            buffer=a.data
        )
        for a in message.arrays
    ]
