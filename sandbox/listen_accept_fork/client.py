from __future__ import print_function

import os
import socket
import logging

def info(message, *args):
    logging.info('%d:' + message, os.getpid(), *args)

def main():
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s: %(message)s',
        level=logging.INFO
    )

    info('client')

if __name__ == '__main__':
    main()


