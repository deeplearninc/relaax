from __future__ import print_function

import signal
import sys
import time

import relaax.common.metrics

from ..common import algorithm_loader


def run(yaml, bind, saver, metrics):
    algorithm = algorithm_loader.load(yaml['path'])

    parameter_server = algorithm.ParameterServer(
        config=algorithm.Config(yaml),
        saver=saver,
        metrics=_Metrics(metrics, lambda: parameter_server.global_t())
    )

    print('looking for checkpoint in %s ...' % parameter_server.checkpoint_location())
    if parameter_server.restore_latest_checkpoint():
        print('checkpoint restored from %s' % parameter_server.checkpoint_location())
        print("global_t is %d" % parameter_server.global_t())

    def stop_server(_1, _2):
        print('')
        _save(parameter_server)
        parameter_server.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)

    # keep the server or else GC will stop it
    server = algorithm.BridgeControl().start_parameter_server(bind, parameter_server.bridge())

    last_global_t = parameter_server.global_t()
    last_activity_time = None
    while True:
        time.sleep(1)

        global_t = parameter_server.global_t()
        if global_t == last_global_t:
            if last_activity_time is not None and time.time() >= last_activity_time + 10:
                _save(parameter_server)
                last_activity_time = None
        else:
            last_activity_time = time.time()

            last_global_t = global_t
            print("global_t is %d" % global_t)


def _log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)


def _save(parameter_server):
    print(
        'checkpoint %d is saving to %s ...' %
        (parameter_server.global_t(), parameter_server.checkpoint_location())
    )
    parameter_server.save_checkpoint()
    print('done')


class _Metrics(relaax.common.metrics.Metrics):
    def __init__(self, metrics, global_t):
        self._metrics = metrics
        self._global_t = global_t

    def scalar(self, name, y, x=None):
        if x is None:
            x = self._global_t()
        self._metrics.scalar(name, y, x=x)
