from __future__ import print_function

import sys
sys.path.append('../../pkg')
sys.path.append('../../server')

import argparse
import logging
import time
import signal
import ruamel.yaml

import algorithms.a3c.bridge
import algorithms.a3c.master
import algorithms.a3c.params

import saver.fs_saver
import saver.s3_saver


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
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='TensorFlow checkpoint directory')
    parser.add_argument('--checkpoint-aws-s3', nargs=2, type=str, default=None, help='AWS S3 bucket and key for TensorFlow checkpoint')
    parser.add_argument('--aws-keys', type=str, default=None, help='YAML file containing AWS access and secret keys')
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(message)s',
        level=log_level
    )

    master = algorithms.a3c.master.Master(
        params=algorithms.a3c.params.Params(_load_yaml(args.params)),
        saver=_saver(args)
    )

    if master.restore_latest_checkpoint():
        print('checkpoint restored from %s' % master.checkpoint_place())

    def stop_server(_1, _2):
        master.save_checkpoint()
        print('checkpoint saved to %s' % master.checkpoint_place())
        master.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop_server)

    # keep the server or else GC will stop it
    server = algorithms.a3c.bridge.start_server(args.bind, master.service())

    last_global_t = None
    while True:
        time.sleep(1)
        global_t = master.global_t()
        if global_t != last_global_t:
            last_global_t = global_t
            print("global_t is %d" % global_t)


def _load_yaml(path):
    with open(path, 'r') as f:
        return ruamel.yaml.load(f, Loader=ruamel.yaml.Loader)

def _saver(args):
    if args.checkpoint_dir is not None:
        return saver.fs_saver.FsSaver(args.checkpoint_dir)
    if args.checkpoint_aws_s3 is not None:
        if args.aws_keys is None:
            aws_access_key = None
            aws_secret_key = None
        else:
            aws_keys = _load_yaml(args.aws_keys)
            aws_access_key = aws_keys['access']
            aws_secret_key = aws_keys['secret']

        return saver.s3_saver.S3Saver(
            *args.checkpoint_aws_s3,
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key
        )

if __name__ == '__main__':
    main()
