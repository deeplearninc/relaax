import tensorflow as tf

def cluster(n_worker=None):
    all_workers = ['localhost:2223', 'localhost:2224']
    workers = all_workers
    if n_worker is not None:
        workers = {n_worker: all_workers[n_worker]}
    return tf.train.ClusterSpec({
        'ps': ['localhost:2222'],
        'worker': workers
    })

def params():
    with tf.device('/job:ps/task:0'):
        return tf.Variable(0., name='sum')
