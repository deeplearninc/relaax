from __future__ import print_function

import sys
sys.path.append('../../pkg')
sys.path.append('../../server')

import logging

import algorithms.a3c.params
import algorithms.a3c.master
import algorithms.a3c.worker
import loop.socket_loop


class _AgentFactory(object):
    def __init__(self, params, master, log_dir):
        self._params = params
        self._master = master
        self._log_dir = log_dir
        self._n_worker = 0

    def __call__(self):
        return algorithms.a3c.worker.Factory(
            params=self._params,
            master=algorithms.a3c.master.Stub(self._master),
            log_dir='%s/worker_%d' % (self._log_dir, self._n_worker)
        )()


def main():
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(message)s',
        level=logging.INFO
    )

    params = algorithms.a3c.params.Params()
    lstm_str = ''
    if params.use_LSTM:
        lstm_str = 'lstm_'

    loop.socket_loop.run_agents(
        'localhost:7000',
        _AgentFactory(
            params=params,
            master='localhost:50051',
            log_dir='logs/boxing_a3c_%s%dthreads' % (lstm_str, 1)
        )
    )


if __name__ == '__main__':
    main()
