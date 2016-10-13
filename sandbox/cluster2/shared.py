import numpy as np
import tensorflow as tf


PS_INDEX = 0


def worker_index(n_worker):
    if n_worker >= PS_INDEX:
        return n_worker + 1
    return n_worker


def cluster(n_worker=None):
    all_workers = ['localhost:2222', 'localhost:2223', 'localhost:2224']
    workers = all_workers
    # workers = {PS_INDEX: all_workers[PS_INDEX]}
    if n_worker is not None:
        workers = {i: all_workers[i] for i in [PS_INDEX, worker_index(n_worker)]}
    return tf.train.ClusterSpec({
        #'ps': ['localhost:2222'],
        'worker': workers
    })


def ps():
    return tf.train.Server(cluster(), job_name='worker', task_index=PS_INDEX)


def worker(n_worker):
    return tf.train.Server(cluster(n_worker), job_name='worker', task_index=worker_index(n_worker))


def params():
    with tf.device('/job:worker/task:%d' % PS_INDEX):
        return tf.Variable(0., name='sum')


def computation(n_worker, sum, inc):
    with tf.device('/job:worker/task:%d' % worker_index(n_worker)):
        increment = sum.assign(sum + inc)

        x_data = np.random.rand(10000000).astype(np.float32)
        y_data = x_data * .1 + .3 * np.random.normal(scale=.1, size=len(x_data))

        W = tf.Variable(tf.random_uniform([1], .0, 1.))
        b = tf.Variable(tf.zeros([1]))
        y = W * x_data + b

        loss = tf.reduce_mean(tf.square(y - y_data))

        train = tf.train.AdagradOptimizer(0.01).minimize(loss)

        vars = [(tf.is_variable_initialized(var), tf.initialize_variables([var])) for var in tf.all_variables()]

    return vars, loss, train, increment
