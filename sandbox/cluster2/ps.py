import time
import signal
import sys
import tensorflow as tf

import shared


def signal_handler(signal, frame):
    sys.exit(0)


def main():

    server = shared.ps()

    signal.signal(signal.SIGINT, signal_handler)

    sum = shared.params()

    init = tf.initialize_all_variables()

    with tf.Session(server.target) as sess:
        sess.run(init)
        v = None
        while True:
            time.sleep(1)
            vv = sess.run(sum)
            if vv != v:
                v = vv
                print v


if __name__ == '__main__':
    main()
