import time
import tensorflow as tf

cluster = tf.train.ClusterSpec({
    'ps': ['localhost:2222'],
    'worker': ['localhost:2223']
})

with tf.device('/job:ps/task:0'):
    sum = tf.Variable(0)
    init = tf.initialize_all_variables()

server = tf.train.Server(cluster, job_name='ps', task_index=0)
with tf.Session(server.target) as sess:
    sess.run(init)
    while True:
        time.sleep(1)
        print 'UGU'
        print sess.run(sum)

