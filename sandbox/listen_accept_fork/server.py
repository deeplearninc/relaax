from __future__ import print_function

import os
import socket
import logging


def info(message, *args):
    logging.info('%d:' + message, os.getpid(), *args)


def handle_connection(c, address):
    info('%s: start', address)
    while True:
        data = c.recv(1024)
        if data is None:
            break
        s = data.decode('utf-8')
        info('%s: received %s', address, repr(s))
    info('%s: stop', address)


def main():
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s: %(message)s',
        level=logging.INFO
    )

    s = socket.socket()
    s.bind('localhost', 7000)
    s.listen(100)

    while True:
        c, address = s.accept()
        pid = os.fork()
        if pid == 0:
            s.close()
            handle_connection(c, address)
            c.close()
            break
        c.close()

    s.close()

if __name__ == '__main__':
    main()

