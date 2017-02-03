from __future__ import print_function

import signal
import sys
import time

import relaax.common.metrics

from ..common import algorithm_loader


def run(yaml, bind, saver, intervals, metrics):
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

    last_saved_global_t = parameter_server.global_t()

    def stop_server(_1, _2):
        print('')
        _save(parameter_server, last_saved_global_t)
        parameter_server.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)

    # keep the server or else GC will stop it
    server = algorithm.BridgeControl().start_parameter_server(bind, parameter_server.bridge())

    intervals_ = []
    if intervals['checkpoint_time_interval'] is not None:
        intervals_.append(_Interval(
            intervals['checkpoint_time_interval'],
            lambda: time.time()
        ))
    if intervals['checkpoint_global_step_interval'] is not None:
        intervals_.append(_Interval(
            intervals['checkpoint_global_step_interval'],
            lambda: parameter_server.global_t()
        ))

    last_activity_time = None
    while True:
        time.sleep(10)

        # do not interrupt loop on first True value
        # we need to update all intervals
        save = False
        for i in intervals_:
            if i.check():
                save = True
        if save:
            print('SAVE')
            _save(parameter_server, last_saved_global_t)
            last_saved_global_t = parameter_server.global_t()


def _save(parameter_server, last_saved_global_t):
    global_t = parameter_server.global_t()
    if global_t == last_saved_global_t:
        return

    print(
        'checkpoint %d is saving to %s ...' %
        (global_t, parameter_server.checkpoint_location())
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


class _Interval(object):
    def __init__(self, interval, counter):
        self._interval = interval
        self._counter = counter
        self._target = counter() + interval

    def check(self):
        counter = self._counter()
        if self._target <= counter:
            self._target = counter + self._interval
            return True
        return False
