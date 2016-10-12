import tensorflow as tf

def params():
    with tf.device('/job:ps/task:0'):
        return tf.Variable(0., name='sum')
