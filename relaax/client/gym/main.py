from __future__ import print_function

import argparse
import logging
import os
import sys

sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    '../../..'
)))

import relaax.client.gym.client


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

    relaax.client.gym.client.run(
        rlx_server=args.rlx_server,
        env=args.env,
        seed=args.seed
    )


main()
