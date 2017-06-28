from __future__ import absolute_import
from builtins import object
import logging
import os
import traceback

from .rlx_config import options
from relaax.common import profiling

log = logging.getLogger(__name__)


class RLXWorker(object):

    @classmethod
    def run(cls, socket, address):
        try:
            if hasattr(options.relaax_rlx_server, 'profile_dir'):
                profiling.set_handlers([profiling.FileHandler(os.path.join(
                    options.relaax_rlx_server.profile_dir, 'rlx_%d.txt' % os.getpid()))])
                profiling.enable(True)
            log.debug('Running worker on connection %s:%d' % address)
            options.protocol.adoptConnection(socket, address)

        except Exception:
            log.error("Something crashed in the worker")
            log.error(traceback.format_exc())
