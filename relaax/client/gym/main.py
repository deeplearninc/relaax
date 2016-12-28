from __future__ import print_function

import argparse
import logging

from .client import run


def main():
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s: %(message)s',
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--rlx-server', type=str, default=None, help='RLX server address (host:port)')
    parser.add_argument('--env', type=str, help='Name of the gym\'s environment')
    parser.add_argument('--seed', type=int, default=None, help='Seed for random generator')
    args = parser.parse_args()

    run(
        rlx_server=args.rlx_server,
        env=args.env,
        seed=args.seed
    )
