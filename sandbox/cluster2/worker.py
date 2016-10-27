import argparse
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

    server = shared.worker(args.task)

    sum = shared.params()

    vars, loss, train, increment = shared.computation(args.task, sum, args.increment)

    tf.train.SummaryWriter('logs/%d' % args.task, graph=tf.get_default_graph())

    with tf.Session(server.target) as sess:
        for var in vars:
            if not sess.run(var[0]):
                sess.run(var[1])

        while True:
            for _ in xrange(10):
                sess.run(train)
            sess.run(increment)
            print sess.run(loss)

if __name__ == '__main__':
    main()
