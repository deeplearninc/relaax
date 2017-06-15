from __future__ import absolute_import
from builtins import object

import errno
import socket
import logging
import multiprocessing as mp

from .rlx_worker import RLXWorker

log = logging.getLogger(__name__)


class RLXPort(object):

    @classmethod
    def listen(cls, server_address):
        cls.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            cls.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            cls.listener.bind(server_address)
            cls.listener.listen(100)
            log.debug("Started and listening on %s:%d" % server_address)

            while True:
                try:
                    connection, address = cls.listener.accept()
                    log.debug("Accepted connection from %s:%s, starting worker" % address)
                except socket.error as e:
                    if cls.handle_accept_socket_exeption(e):
                        continue
                    raise
                except KeyboardInterrupt:
                    # Swallow KeyboardInterrupt
                    break

                try:
                    p = mp.Process(target=cls.start_worker, args=(connection, address))
                    p.start()
                except Exception as e:
                    log.critical('Can\'t start child process {}: {}'.format(address, str(e)))
                    connection.close()
                    log.debug('Closing connection %s:%d' % address)
        finally:
            log.debug('Closing listener')
            cls.listener.close()

    @classmethod
    def start_worker(cls, connection, address):
        try:
            RLXWorker.run(connection, address)
        finally:
            log.debug('Closing connection %s:%d' % address)
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except socket.error as e:
                # we don't care if the socket is already closed;
                # this will often be the case if client closed connection first
                if e.errno != errno.ENOTCONN:
                    raise
            finally:
                connection.close()

    @classmethod
    def handle_accept_socket_exeption(cls, error):
        if error.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            # Try again
            return True  # continue accept loop
        elif error.errno == errno.EPERM:
            # Netfilter on Linux may have rejected the
            # connection, but we get told to try to accept()
            # anyway.
            return True  # continue accept loop
        elif error.errno in (errno.EMFILE, errno.ENOBUFS, errno.ENFILE,
                             errno.ENOMEM, errno.ECONNABORTED):
            # Linux gives EMFILE when a process is not allowed to
            # allocate any more file descriptors.  *BSD and Win32
            # give (WSA)ENOBUFS.  Linux can also give ENFILE if the
            # system is out of inodes, or ENOMEM if there is
            # insufficient memory to allocate a new dentry.
            # ECONNABORTED is documented as possible on all
            # relevant platforms (Linux, Windows, macOS, and the
            # BSDs) but occurs only on the BSDs.  It occurs when a
            # client sends a FIN or RST after the server sends a
            # SYN|ACK but before application code calls accept(2).
            # On Linux, calling accept(2) on such a listener
            # returns a connection that fails as though the it were
            # terminated after being fully established.  This
            # appears to be an implementation choice (see
            # inet_accept in inet/ipv4/af_inet.c).  On macOS X,
            # such a listener is not considered readable, so
            # accept(2) will never be called.  Calling accept(2) on
            # such a listener, however, does not return at all.
            log.error("Could not accept new connection (%s)" % error.strerror)
        return False  # break accept loop
