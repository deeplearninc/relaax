from __future__ import print_function

import itertools
import logging
import os
import psutil
import signal
import socket

from .worker import Worker
from ..common import algorithm_loader


def run(bind_address, yaml, parameter_server_url, timeout):
    algorithm = algorithm_loader.load(yaml['path'])
    _run_agents(
        bind_address=bind_address,
        agent_factory=_get_factory(
            algorithm=algorithm,
            yaml=yaml,
            parameter_server_url=parameter_server_url
        ),
        timeout=timeout
    )


def _get_factory(algorithm, yaml, parameter_server_url):
    config = algorithm.Config(yaml)
    return lambda n_agent: algorithm.Agent(
        config=config,
        parameter_server=algorithm.BridgeControl().parameter_server_stub(parameter_server_url)
    )


def _run_agents(bind_address, agent_factory, timeout):
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)

    socket_ = socket.socket()
    try:
        _info('listening %s', bind_address)

        socket_.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        socket_.bind(_parse_address(bind_address))
        socket_.listen(100)

        n_agent = 0
        while True:
            connection, address = socket_.accept()
            try:
                _debug('accepted %s from %s', str(connection), str(address))
                available = _available_memory()
                required = _memory_per_child()
                if required is None:
                    _info('memory %.3f, None' % available)
                else:
                    _info('memory %.3f, %.3f' % (available, required))
                if required is not None and available < required:
                    _warning(
                        'Cannot start new child: available memory (%.3f) is less than memory per child (%.3f)' %
                        (available, required)
                    )
                else:
                    pid = None
                    try:
                        pid = os.fork()
                    except OSError as e:
                        _warning('{} : {}'.format(bind_address, e.message))
                    if pid == 0:
                        socket_.close()
                        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        Worker(agent_factory, timeout, n_agent, connection, address).run()
                        break
            finally:
                connection.close()
            n_agent += 1
    finally:
        socket_.close()


def _debug(message, *args):
    logging.debug('%d:' + message, os.getpid(), *args)


def _info(message, *args):
    logging.info('%d:' + message, os.getpid(), *args)


def _parse_address(address):
    host, port = address.split(':')
    return host, int(port)


def _available_memory():
    vm = psutil.virtual_memory()
    return 100 * float(vm.available) / vm.total


def _memory_per_child():
    process = psutil.Process(os.getpid())
    mem = sum(itertools.imap(
        _process_memory,
        process.children(recursive=True)
    ))
    n = len(process.children(recursive=False))
    if n == 0:
        return None
    return mem / n


def _process_memory(process):
    try:
        return process.memory_percent()
    except psutil.NoSuchProcess:
        return 0


def _warning(message, *args):
    logging.warning('%d:' + message, os.getpid(), *args)
