import time
import signal
import sys
import tensorflow as tf

import shared


def main():

    server = shared.ps()

    signal.signal(signal.SIGINT, lambda _1, _2: sys.exit(0))

    while True:
        time.sleep(1)
        print 'UGU'


if __name__ == '__main__':
    main()
