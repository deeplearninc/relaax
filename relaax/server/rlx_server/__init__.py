from __future__ import print_function

import argparse
import logging
import ruamel.yaml

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

    socket_loop.run_agents(
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
