import numpy as np
import tensorflow as tf


def cluster(n_worker=None):
    all_workers = ['localhost:%d' % (2223 + i) for i in xrange(2)]
    workers = all_workers
    if n_worker is not None:
        workers = {n_worker: all_workers[n_worker]}
    return tf.train.ClusterSpec({
        'ps': ['localhost:2222'],
        'worker': workers
    })


def ps():
    return tf.train.Server(cluster(), job_name='ps', task_index=0)


def worker(n_worker):
    return tf.train.Server(cluster(n_worker), job_name='worker', task_index=n_worker)


def ps_device():
    return '/job:ps/task:0'


def worker_device(n_worker):
    return '/job:worker/task:%d' % n_worker
