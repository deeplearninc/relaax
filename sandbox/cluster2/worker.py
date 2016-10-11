import argparse
import tensorflow as tf
import time

import params

def parse_args():
    parser = argparse.ArgumentParser(description='It\'s worker to increment sum.')
    parser.add_argument('increment', metavar='N', type=float, help='increment value')
    parser.add_argument('times', metavar='T', type=int, help='times to repeat increments')
    parser.add_argument('interval', metavar='I', type=float, help='interval between increments')

    return parser.parse_args()


def main():
    args = parse_args()

    sum = params.params()

    increment = sum.assign(sum + args.increment)

    with tf.Session('grpc://localhost:2222') as sess:
        for i in xrange(args.times):
            if i > 0:
                time.sleep(args.interval)
            sess.run(increment)

if __name__ == '__main__':
    main()
