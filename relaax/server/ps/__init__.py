from __future__ import print_function

import argparse
import logging
import ruamel.yaml
import signal
import sys
import time

from ...algorithms import da3c
from ..common.saver import fs_saver, s3_saver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log-level',
        type=str,
        default='WARNING',
        choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
        help='logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)'
    )
    parser.add_argument('--config', type=str, default=None, help='configuration YAML file')
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

    ps = da3c.Ps(
        config=da3c.Config(_load_yaml(args.config)),
        saver=_saver(args)
    )

    print('looking for checkpoint in %s ...' % ps.checkpoint_place())
    if ps.restore_latest_checkpoint():
        print('checkpoint restored from %s' % ps.checkpoint_place())
        print("global_t is %d" % ps.global_t())

    def stop_server(_1, _2):
        print('')
        _save(ps)
        ps.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, stop_server)

    # keep the server or else GC will stop it
    server = da3c.start_ps(args.bind, _Service(ps))

    last_global_t = ps.global_t()
    last_activity_time = None
    while True:
        time.sleep(1)

        global_t = ps.global_t()
        if global_t == last_global_t:
            if last_activity_time is not None and time.time() >= last_activity_time + 10:
                _save(ps)
                last_activity_time = None
        else:
            last_activity_time = time.time()

            last_global_t = global_t
            print("global_t is %d" % global_t)


class _Service(da3c.PsService):
    def __init__(self, ps):
        self.increment_global_t = ps.increment_global_t
        self.apply_gradients = ps.apply_gradients
        self.get_values = ps.get_values


def _log_uniform(lo, hi, rate):
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    v = log_lo * (1 - rate) + log_hi * rate
    return math.exp(v)


def _load_yaml(path):
    with open(path, 'r') as f:
        return ruamel.yaml.load(f, Loader=ruamel.yaml.Loader)


def _saver(args):
    if args.checkpoint_dir is not None:
        return fs_saver.FsSaver(args.checkpoint_dir)
    if args.checkpoint_aws_s3 is not None:
        if args.aws_keys is None:
            aws_access_key = None
            aws_secret_key = None
        else:
            aws_keys = _load_yaml(args.aws_keys)
            aws_access_key = aws_keys['access']
            aws_secret_key = aws_keys['secret']

        return s3_saver.S3Saver(
            *args.checkpoint_aws_s3,
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key
        )


def _save(ps):
    print('checkpoint %d is saving to %s ...' % (ps.global_t(), ps.checkpoint_place()))
    ps.save_checkpoint()
    print('done')
