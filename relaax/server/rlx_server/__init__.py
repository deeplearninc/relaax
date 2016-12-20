from __future__ import print_function

import argparse
import logging
import ruamel.yaml

from ...algorithms import da3c
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

    config_ = da3c.Config(yaml)

    socket_loop.run_agents(
        bind_address=args.bind,
        agent_factory=_get_factory(
            config=config_,
            parameter_server=args.parameter_server,
            log_dir=args.log_dir
        ),
        timeout=args.timeout
    )


def _get_factory(config, parameter_server, log_dir):
    return lambda n_agent: da3c.Agent(
        config=config,
        parameter_server=da3c.ParameterServerStub(parameter_server),
        log_dir='%s/worker_%d' % (log_dir, n_agent)
    )
