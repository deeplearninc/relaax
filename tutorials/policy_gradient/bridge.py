from __future__ import print_function

import concurrent
import grpc
import itertools
import numpy

import relaax.algorithm_base.bridge_base

from . import bridge_pb2


class BridgeControl(relaax.algorithm_base.bridge_base.BridgeControlBase):
    def parameter_server_stub(self, parameter_server_url):
        return _Stub(parameter_server_url)

    def start_parameter_server(self, address, service):
        server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=1))
        bridge_pb2.add_ParameterServerServicer_to_server(_Servicer(service), server)
        server.add_insecure_port(address)
        server.start()
        return server


class _Stub(relaax.algorithm_base.bridge_base.BridgeBase):
    def __init__(self, parameter_server):
        self._stub = bridge_pb2.ParameterServerStub(grpc.insecure_channel(parameter_server))
        self._metrics = _Metrics(self._stub)


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


def _build_ndarray_message(array):
    return bridge_pb2.NdArray(
        dtype=str(array.dtype),
        shape=array.shape,
        data=array.tobytes()
    )


def _parse_ndarray_message(message):
    return numpy.ndarray(
        shape=message.shape,
        dtype=numpy.dtype(message.dtype),
        buffer=message.data
    )
