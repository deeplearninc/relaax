from __future__ import print_function

import os
import socket
import logging

def info(message, *args):
    logging.info('%d:' + message, os.getpid(), *args)

def handle_connection():
    pass

def main():
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s: %(message)s',
        level=logging.INFO
    )
    os.fork()
    info('server')
    pass

if __name__ == '__main__':
    main()

