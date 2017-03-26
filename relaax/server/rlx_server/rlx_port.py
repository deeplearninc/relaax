import os
import errno
import socket
import logging

from rlx_worker import RLXWorker

log = logging.getLogger(__name__)


class RLXPort():

    @classmethod
    def listen(cls, server_address):
        cls.listener = socket.socket()
        try:
            cls.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            cls.listener.bind(server_address)
            cls.listener.listen(100)
            log.debug("Started and listening on %s:%d" % server_address)

            while True:
                try:
                    connection, address = cls.listener.accept()
                    log.debug("Accepted connection, starting worker")
                except socket.error as e:
                    if cls.handle_accept_socket_exeption(e):
                        continue
                    raise
                except KeyboardInterrupt:
                    # Swallow KeyboardInterrupt
                    break

                try:
                    pid = None
                    try:
                        pid = os.fork()
                    except OSError as e:
                        log.critical('OSError {}: {}'.format(address, e.message))

                    if pid == 0:
                        RLXWorker.run(connection, address)
                        break

                finally:
                    log.debug("Closing accepted connection")
                    connection.close()
        finally:
            log.debug('Closing listening socket')
            cls.listener.close()

    @classmethod
    def handle_accept_socket_exeption(cls, error):
        if error[0] in (errno.EWOULDBLOCK, errno.EAGAIN):
            # Try again
            return True  # continue accept loop
        elif error[0] == errno.EPERM:
            # Netfilter on Linux may have rejected the
            # connection, but we get told to try to accept()
            # anyway.
            return True  # continue accept loop
        elif error[0] in (errno.EMFILE, errno.ENOBUFS, errno.ENFILE,
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
            log.error("Could not accept new connection (%s)" % error[1])
        return False  # break accept loop
