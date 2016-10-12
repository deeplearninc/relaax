import argparse
import numpy as np
import tensorflow as tf
import time

import shared

def parse_args():
    parser = argparse.ArgumentParser(description='It\'s worker to increment sum.')
    parser.add_argument('task', metavar='T', type=int, help='task number')
    parser.add_argument('increment', metavar='N', type=float, help='increment value')

    return parser.parse_args()


def main():
    args = parse_args()

    server = tf.train.Server(shared.cluster(), job_name='ps', task_index=args.task)

    sum = shared.params()

    with tf.device('/job:ps/task:%d' % args.task):
        increment = sum.assign(sum + args.increment)

        x_data = np.random.rand(10000000).astype(np.float32)
        y_data = x_data * .1 + .3 * np.random.normal(scale=.1, size=len(x_data))

        W = tf.Variable(tf.random_uniform([1], .0, 1.))
        b = tf.Variable(tf.zeros([1]))
        y = W * x_data + b

        loss = tf.reduce_mean(tf.square(y - y_data))

        train = tf.train.AdagradOptimizer(0.01).minimize(loss)

        vars = [(tf.is_variable_initialized(var), tf.initialize_variables([var])) for var in tf.all_variables()]

        tf.scalar_summary("loss", loss)
        summary = tf.merge_all_summaries()

    sw = tf.train.SummaryWriter('logs', graph=tf.get_default_graph())

    with tf.Session(server.target) as sess:
        for var in vars:
            if not sess.run(var[0]):
                sess.run(var[1])

        while True:
            sw.add_summary(sess.run(summary))

            sess.run(train)
            sess.run(increment)
            print sess.run(loss)

if __name__ == '__main__':
    main()
