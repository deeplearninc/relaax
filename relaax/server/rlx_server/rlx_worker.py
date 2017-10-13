from __future__ import absolute_import
from builtins import object
import logging
import os
import traceback

from relaax.server.common.algorithm_loader import AlgorithmLoader
from .rlx_config import options
from relaax.common import profiling

log = logging.getLogger(__name__)


class RLXWorker(object):

    @staticmethod
    def load_algorithm():
        try:
            options.Agent = AlgorithmLoader.load_agent(
                options.algorithm_path, options.algorithm_name)

        except Exception:
            log.critical("Can't load algorithm module")
            raise

    @staticmethod
    def load_protocol():
        name = options.protocol_name

        if name == 'rawsocket':
            log.debug("Loading protocol over raw socket")
            from .rlx_protocol.rawsocket import rlx_protocol as protocol
            options.protocol = protocol
        elif name == 'twisted':
            log.debug("Loading protocol on twisted")
            from .rlx_protocol.twisted import rlx_protocol as protocol
            options.protocol = protocol
        else:
            raise Exception("Can't load protocol %s" % str(name))

    @classmethod
    def run(cls, socket, address):
        try:
            if hasattr(options.relaax_rlx_server, 'profile_dir'):
                profiling.set_handlers([profiling.FileHandler(os.path.join(
                    options.relaax_rlx_server.profile_dir, 'rlx_%d.txt' % os.getpid()))])
                profiling.enable(True)
            cls.load_protocol()
            cls.load_algorithm()
            log.debug('Running worker on connection %s:%d' % address)
            options.protocol.adoptConnection(socket, address)

        except Exception as e:
            log.error("Error while running worker on connection %s : %s" % (address, str(e)))
            log.debug(traceback.format_exc())
