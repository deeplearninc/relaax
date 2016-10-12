import tensorflow as tf

def cluster():
    return tf.train.ClusterSpec({'ps': ['localhost:2222', 'localhost:2223', 'localhost:2224']})

def params():
    with tf.device('/job:ps/task:0'):
        return tf.Variable(0., name='sum')
