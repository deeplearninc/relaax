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


class _Stub(object):
    def __init__(self, parameter_server):
        self._stub = bridge_pb2.ParameterServerStub(grpc.insecure_channel(parameter_server))
        self._metrics = _Metrics(self._stub)

    def get_global_t(self):
        return self._stub.GetGlobalT(bridge_pb2.NullMessage()).g

    def get_filter_state(self):
        state = self._stub.GetFilterState(bridge_pb2.NullMessage())
        return state.n, _parse_ndarray_message(state.mean), _parse_ndarray_message(state.std)

    def wait_for_iteration(self):
        return self._stub.WaitForIteration(bridge_pb2.NullMessage()).n_iter

    def send_experience(self, n_iter, paths, length):
        self._stub.SendExperience(bridge_pb2.Experience(
            n_iter=n_iter,
            observation=itertools.imap(_build_ndarray_message, paths["observation"]),
            action=itertools.imap(_build_ndarray_message, paths["action"]),
            prob=itertools.imap(_build_ndarray_message, paths["prob"]),
            reward=paths["reward"],
            terminated=paths["terminated"],
            length=length,
            diff=bridge_pb2.FilterState(
                n=length,
                mean=_build_ndarray_message(paths["filter_diff"][1]),
                std=_build_ndarray_message(paths["filter_diff"][2])
            )
        ))

    def receive_weights(self, n_iter):
        return itertools.imap(
            _parse_ndarray_message,
            self._stub.ReceiveWeights(bridge_pb2.NIter(n_iter=n_iter))
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

    def GetGlobalT(self, request, context):
        return bridge_pb2.Step(g=long(self._service.get_global_t()))

    def GetFilterState(self, request, context):
        n, mean, std = self._service.get_filter_state()
        return bridge_pb2.FilterState(n=n,
                                      mean=_build_ndarray_message(mean),
                                      std=_build_ndarray_message(std)
                                      )

    def WaitForIteration(self, request, context):
        return bridge_pb2.NIter(n_iter=self._service.wait_for_iteration())

    def SendExperience(self, request, context):
        self._service.send_experience(request.n_iter, {
            'observation': map(_parse_ndarray_message, request.observation),
            'action': map(_parse_ndarray_message, request.action),
            'prob': map(_parse_ndarray_message, request.prob),
            'reward': numpy.array(request.reward),
            'terminated': request.terminated,
            'filter_diff': (request.length,
                            _parse_ndarray_message(request.diff.mean),
                            _parse_ndarray_message(request.diff.std))
        }, request.length)
        return bridge_pb2.NullMessage()

    def ReceiveWeights(self, request, context):
        for weights in self._service.receive_weights(request.n_iter):
            yield _build_ndarray_message(weights)

    def StoreScalarMetric(self, request, context):
        x = None
        if request.HasField('x'):
            x = request.x.value
        self._service.metrics().scalar(name=request.name, y=request.y, x=x)
        return bridge_pb2.NullMessage()


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
