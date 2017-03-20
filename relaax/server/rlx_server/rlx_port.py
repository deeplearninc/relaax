import os
import sys
import socket
import logging
log = logging.getLogger(__name__)

from rlx_worker import RLXWorker

class RLXPort():

    @classmethod
    def listen(self,server_address):
        self.listener = socket.socket()
        try:
            self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.listener.bind(server_address)
            self.listener.listen(100)
            log.debug("Started and listening on %s:%d" % server_address)

            while True:
                try:    
                    connection, address = self.listener.accept()
                    log.debug("Accepted connection, starting worker")
                except socket.error as e:
                    if handle_accept_socket_exeption(e):
                        connection
                    raise
                except KeyboardInterrupt:
                    # Swallow KeyboardInterrupt
                    break

                try:
                    pid = None
                    try:
                        pid = os.fork()
                    except OSError as e:
                        log.critical('OSError {} : {}'.format(server_address, e.message))
                    if pid == 0:
                        RLXWorker.run(connection,address)
                        break

                finally:
                    log.debug("Closing accepted connection")
                    connection.close()
        finally:
            log.debug('Closing listening socket')
            self.listener.close()

    @classmethod
    def handle_accept_socket_exeption(error):
        if error.args[0] in (EWOULDBLOCK, EAGAIN):
            # Try again
            return True # continue accept loop
        elif error.args[0] == EPERM:
            # Netfilter on Linux may have rejected the
            # connection, but we get told to try to accept()
            # anyway.
            return True # continue accept loop
        elif error.args[0] in (EMFILE, ENOBUFS, ENFILE, ENOMEM, ECONNABORTED):
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
            log.error("Could not accept new connection (%s)" % (
                errorcode[error.args[0]],))
        return False # break accept loop

