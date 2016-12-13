from __future__ import print_function

import sys
sys.path.append('../../pkg')
sys.path.append('../../server')

import argparse
import logging
import ruamel.yaml

import algorithms.a3c.agent
import algorithms.a3c.bridge
import algorithms.a3c.params
import loop.socket_loop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log-level',
        type=str,
        default='WARNING',
        choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
        help='logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)'
    )
    parser.add_argument('--params', type=str, default=None, help='parameters YAML file')
    parser.add_argument('--bind', type=str, default=None, help='address to serve (host:port)')
    parser.add_argument('--master', type=str, default=None, help='master address (host:port)')
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

    with open(args.params, 'r') as f:
        yaml = ruamel.yaml.load(f, Loader=ruamel.yaml.Loader)

    params = algorithms.a3c.params.Params(yaml)

    loop.socket_loop.run_agents(
        bind_address=args.bind,
        agent_factory=_get_factory(
            params=params,
            master=args.master,
            log_dir=args.log_dir
        ),
        timeout=args.timeout
    )


def _get_factory(params, master, log_dir):
    return lambda n_agent: algorithms.a3c.agent.Agent(
        params=params,
        master=algorithms.a3c.bridge.MasterStub(master),
        log_dir='%s/worker_%d' % (log_dir, n_agent)
    )


if __name__ == '__main__':
    main()
