import time
import signal
import sys
import tensorflow as tf

import params

def signal_handler(signal, frame):
    sys.exit(0)

def main():
    server = tf.train.Server(
        tf.train.ClusterSpec({'ps': ['localhost:2222']}),
        job_name='ps',
        task_index=0
    )

    signal.signal(signal.SIGINT, signal_handler)

    sum = params.params()

    init = tf.initialize_all_variables()

    with tf.Session('grpc://localhost:2222') as sess:
        sess.run(init)
        while True:
            time.sleep(1)
            print sess.run(sum)

if __name__ == '__main__':
    main()


