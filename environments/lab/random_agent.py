from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging

from .client import run


def main():
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s: %(message)s',
        level=logging.INFO
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', type=str, default=None,
                        help='agent server address (host:port)')
    parser.add_argument('--length', type=int, default=8 * 10 ** 7,
                        help='Number of steps to run the agent')
    parser.add_argument('--width', type=int, default=84,
                        help='Horizontal size of the observations')
    parser.add_argument('--height', type=int, default=84,
                        help='Vertical size of the observations')
    parser.add_argument('--fps', type=int, default=60,
                        help='Number of frames per second')
    parser.add_argument('--runfiles_path', type=str, default=None,
                        help='Set the runfiles path to find DeepMind Lab data')
    parser.add_argument('--level_script', type=str, default='tests/demo_map',
                        help='The environment level script to load')

    args = parser.parse_args()
    if args.runfiles_path:
        deepmind_lab.set_runfiles_path(args.runfiles_path)

    run(
        rlx_server=args.rlx_server,
        level=args.level_script,
        seed=args.seed
    )
