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

    while True:
        time.sleep(1)
        print 'UGU'


if __name__ == '__main__':
    main()
