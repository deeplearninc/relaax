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

    def increment_global_t(self):
        return self._stub.IncrementGlobalT(bridge_pb2.NullMessage()).n

    def apply_gradients(self, gradients):
        self._stub.ApplyGradients(_build_ndarrays_part_messages(gradients))

    def get_values(self):
        return _parse_ndarrays_part_messages(
            self._stub.GetValues(bridge_pb2.NullMessage())
        )

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

    def ApplyGradients(self, request_iterator, context):
        self._service.apply_gradients(
            _parse_ndarrays_part_messages(request_iterator)
        )
        return bridge_pb2.NullMessage()

    def GetValues(self, request, context):
        for part in _build_ndarrays_part_messages(self._service.get_values()):
            yield part

    def StoreScalarMetric(self, request, context):
        x = None
        if request.HasField('x'):
            x = request.x.value
        self._service.metrics().scalar(name=request.name, y=request.y, x=x)
        return bridge_pb2.NullMessage()


def _build_ndarray_part_messages(array):
    # TODO: select more appropriate block size
    block_size = 1024 * 1024

    dtype = str(array.dtype)
    shape = array.shape
    data = array.data
    size = len(data)

    # optimization to avoid extra data copying if array data fits to one block
    # TODO: compare actual performance
    if size <= block_size:
        bytes_ = array.tobytes()
        assert size == len(bytes_)
        yield bridge_pb2.NdArrayPart(
            dtype=dtype,
            shape=shape,
            last_part=True,
            data=bytes_
        )
    else:
        i = 0
        while i < size:
            ii = i + block_size
            yield bridge_pb2.NdArrayPart(
                dtype=dtype,
                shape=shape,
                last_part=ii >= size,
                data=data[i:ii]
            )
            i = ii


def _build_ndarrays_part_messages(arrays):
    for array in arrays:
        for part in _build_ndarray_part_messages(array):
            yield part


def _parse_ndarrays_part_messages(messages):
    data = []
    for message in messages:
        if message.last_part:
            # optimization to avoid extra data copying if array data fits to one block
            # TODO: compare actual performance
            if len(data) == 0:
                yield numpy.ndarray(
                    shape=message.shape,
                    dtype=numpy.dtype(message.dtype),
                    buffer=message.data
                )
            else:
                data.append(message.data)
                yield numpy.ndarray(
                    shape=message.shape,
                    dtype=numpy.dtype(message.dtype),
                    buffer=''.join(data)
                )
                data = []
        else:
            data.append(message.data)
