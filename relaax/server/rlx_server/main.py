from __future__ import print_function

import argparse
import logging
import os
import ruamel.yaml

import relaax.server.rlx_server.server


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
    parser.add_argument('--parameter-server', type=str, default=None, help='parameter server address (host:port)')
    parser.add_argument('--log-dir', type=str, default=None, help='TensorBoard log directory')
    parser.add_argument('--timeout', type=float, default=None, help='Agent stops on game reset after given timeout')
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

    if 'relaax-rlx-server' in yaml:
        cmdl = yaml['relaax-rlx-server']

        if args.log_level is None and '--log-level' in cmdl:
            args.log_level = cmdl['--log-level']
        if args.bind is None and '--bind' in cmdl:
            args.bind = cmdl['--bind']
        if args.parameter_server is None and '--parameter-server' in cmdl:
            args.parameter_server = cmdl['--parameter-server']
        if args.log_dir is None and '--log-dir' in cmdl:
            args.log_dir = cmdl['--log-dir']
        if args.timeout is None:
            if '--timeout' in cmdl:
                args.timeout = cmdl['--timeout']
            else:
                args.timeout = 1000000 # arbitrary large number

    relaax.server.rlx_server.server.run(
        bind_address=args.bind,
        yaml=yaml['algorithm'],
        parameter_server=args.parameter_server,
        log_dir=args.log_dir,
        timeout=args.timeout
    )
