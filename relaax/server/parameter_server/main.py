from __future__ import print_function

import argparse
import logging
import os
import ruamel.yaml
import tensorflow as tf

import relaax.common.metrics
import relaax.server.common.saver.fs_saver
import relaax.server.common.saver.s3_saver
import relaax.server.parameter_server.server


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
    parser.add_argument('--metrics-dir', type=str, default=None, help='TensorBoard metrics directory')
    parser.add_argument('--aws-keys', type=str, default=None, help='YAML file containing AWS access and secret keys')
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(message)s',
        level=log_level
    )

    yaml = _load_yaml(args.config)

    if 'relaax-parameter-server' in yaml:
        cmdl = yaml['relaax-parameter-server']

        if args.log_level is None and '--log-level' in cmdl:
            args.log_level = cmdl['--log-level']
        if args.bind is None and '--bind' in cmdl:
            args.bind = cmdl['--bind']
        if args.checkpoint_dir is None and '--checkpoint-dir' in cmdl:
            args.checkpoint_dir = cmdl['--checkpoint-dir']
        if args.metrics_dir is None and '--metrics-dir' in cmdl:
            args.metrics_dir = cmdl['--metrics-dir']
        if args.checkpoint_aws_s3 is None and '--checkpoint-aws-s3' in cmdl:
            args.checkpoint_aws_s3 = cmdl['--checkpoint-aws-s3']

    relaax.server.parameter_server.server.run(
        yaml=yaml['algorithm'],
        bind=args.bind,
        saver=_saver(args),
        metrics=_TensorFlowMetrics(args.metrics_dir)
    )


def _load_yaml(path):
    with open(path, 'r') as f:
        return ruamel.yaml.load(f, Loader=ruamel.yaml.Loader)


def _saver(args):
    if args.checkpoint_dir is not None:
        return relaax.server.common.saver.fs_saver.FsSaver(args.checkpoint_dir)

    if args.checkpoint_aws_s3 is not None:
        if args.aws_keys is None:
            aws_access_key = None
            aws_secret_key = None
        else:
            aws_keys = _load_yaml(args.aws_keys)
            aws_access_key = aws_keys['access']
            aws_secret_key = aws_keys['secret']

        return relaax.server.common.saver.s3_saver.S3Saver(
            *args.checkpoint_aws_s3,
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key
        )


class _TensorFlowMetrics(relaax.common.metrics.Metrics):
    def __init__(self, metrics_dir):
        self._summaries = {}
        self._graph = tf.Graph()
        self._writer = tf.train.SummaryWriter(metrics_dir, self._graph)
        self._session = tf.Session(graph=self._graph)

    def scalar(self, name, y, x=None):
        with self._graph.as_default():
            if name not in self._summaries:
                placeholder = tf.placeholder(tf.float64)
                self._summaries[name] = (
                    placeholder,
                    tf.scalar_summary(name, placeholder)
                )
            placeholder, summary = self._summaries[name]
            self._writer.add_summary(
                self._session.run(summary, feed_dict={placeholder: y}),
                global_step=x
            )
