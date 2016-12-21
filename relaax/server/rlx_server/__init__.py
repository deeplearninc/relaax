from __future__ import print_function

import argparse
import logging
import os
import psutil
import ruamel.yaml
import signal
import socket
import time

from ..common import algorithm_loader
from ...common.loop import socket_loop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log-level',
        type=str,
        default='WARNING',
        choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
        help='logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)'
    )
    parser.add_argument('--config', type=str, default=None, help='parameters YAML file')
    parser.add_argument('--bind', type=str, default=None, help='address to serve (host:port)')
    parser.add_argument('--parameter_server', type=str, default=None, help='parameter server address (host:port)')
    parser.add_argument('--log-dir', type=str, default=None, help='TensorBoard log directory')
    parser.add_argument('--timeout', type=float, default=120, help='Agent stops on game reset after given timeout')
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(message)s',
        level=log_level
    )

    with open(args.config, 'r') as f:
        yaml = ruamel.yaml.load(f, Loader=ruamel.yaml.Loader)

    algorithm = algorithm_loader.load(yaml['path'])

    _run_agents(
        bind_address=args.bind,
        agent_factory=_get_factory(
            algorithm=algorithm,
            yaml=yaml,
            parameter_server=args.parameter_server,
            log_dir=args.log_dir
        ),
        timeout=args.timeout
    )


def _get_factory(algorithm, yaml, parameter_server, log_dir):
    config = algorithm.Config(yaml)
    return lambda n_agent: algorithm.Agent(
        config=config,
        parameter_server=algorithm.ParameterServerStub(parameter_server),
        log_dir='%s/worker_%d' % (log_dir, n_agent)
    )


def _run_agents(bind_address, agent_factory, timeout):
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)

    socket_ = socket.socket()
    try:
        _info('listening %s', bind_address)

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
                    try:
                        pid = _forkf()
                        if pid == 0:
                            socket_.close()
                            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                            socket_loop.run_agent(
                                connection,
                                address,
                                _AgentServiceFactory(agent_factory(n_agent), timeout)
                            )
                            break
                    except _Failure as e:
                        _warning('{} : {}'.format(bind_address, e.message))
            finally:
                connection.close()
            n_agent += 1
    finally:
        socket_.close()


class _Failure(Exception):
    def __init__(self, message):
        self.message = message


class _AgentService(socket_loop.AgentService):
    def __init__(self, environment_service, agent, timeout):
        self._stop = time.time() + timeout
        self._environment_service = environment_service
        self._agent = agent

    def act(self, state):
        self._environment_service.act(self._agent.act(state))

    def reward_and_reset(self, reward):
        score = self._agent.reward_and_reset(reward)
        if score is None:
            raise _Failure('no answer from agent')
        if time.time() >= self._stop:
            raise _Failure('timeout')
        self._environment_service.reset(score)

    def reward_and_act(self, reward, state):
        action = self._agent.reward_and_act(reward, state)
        if action is None:
            raise _Failure('no answer from agent')
        self._environment_service.act(action)


class _AgentServiceFactory(socket_loop.AgentServiceFactory):
    def __init__(self, agent, timeout):
        self._agent = agent
        self._timeout = timeout

    def __call__(self, environment_service):
        return _AgentService(environment_service, self._agent, self._timeout)


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
    n = 0
    mem = 0
    for child in process.children(recursive=False):
        n += 1
        mem += _process_tree_memory(child)
    if n == 0:
        return None
    return mem / n


def _process_tree_memory(process):
    mem = process.memory_percent()
    for child in process.children(recursive=True):
        mem += child.memory_percent()
    return mem


def _forkf():
    try:
        return os.fork()
    except OSError as e:
        raise _failure('OSError', e)


def _failure(etype, e):
    return _Failure("{}({}): {}".format(etype, e.errno, e.strerror))
