from __future__ import print_function

import logging
import os
import socket
import time

def info(message, *args):
    logging.info('%d:' + message, os.getpid(), *args)

def main():
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s: %(message)s',
        level=logging.INFO
    )

    s = socket.socket()
    s.connect(('localhost', 7000))

    while True:
        time.sleep(1)
        s.send(bytearray('hello', 'utf-8'))
        info('client')

if __name__ == '__main__':
    main()


