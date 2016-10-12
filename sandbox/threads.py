import numpy as np
import signal
import sys
import tensorflow as tf
import threading
import time


class Elapsed(object):
    def __init__(self):
        self._lock = threading.Lock()
        self._last_time = time.time()
        self.factor = 0
        self.sum1 = 0
        self.sum2 = 0
        self.count = 0

    def enter(self):
        self._inc(1)

    def leave(self):
        self._inc(-1)

    def _inc(self, inc):
        with self._lock:
            current_time = time.time()
            elapsed = current_time - self._last_time  

            if self.factor > 0:
                self.sum1 += elapsed
            self.sum2 += elapsed * self.factor
            self.count += 1

            self.factor += inc
            self._last_time = current_time


def graph():
    x_data = np.random.rand(10000).astype(np.float32)
    y_data = x_data * .1 + .3 * np.random.normal(scale=.1, size=len(x_data))

    W = tf.Variable(tf.random_uniform([1], .0, 1.))
    b = tf.Variable(tf.zeros([1]))
    y = W * x_data + b

    loss = tf.reduce_mean(tf.square(y - y_data))

    return tf.train.AdagradOptimizer(0.01).minimize(loss)


def main():
    stop = threading.Event()

    stop_flag = [False]

    def signal_handler(signal, frame):
        stop_flag[0] = True
        stop.set()

    signal.signal(signal.SIGINT, signal_handler)

    for n in xrange(1, 100):
        stop.clear()

        trains = [graph() for i in xrange(n)]
        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            elapsed = Elapsed()
            sess.run(init)

            def thread(id, train):
                while not stop.is_set():
                    #print('%d {' % id)
                    elapsed.enter()
                    sess.run(train)
                    elapsed.leave()
                    #print('%d }' % id)

            threads = [
                threading.Thread(target=thread, args=(i, train))
                for i, train in enumerate(trains)
            ]

            for t in threads:
                t.start()

            time.sleep(5)
            stop.set()

            for t in threads:
                t.join()

            if stop_flag[0]:
                break;

            print('overlap factor is %f for %d threads' % (elapsed.sum2 / elapsed.sum1, n))


if __name__ == '__main__':
    main()
