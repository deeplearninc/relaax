from __future__ import absolute_import
from builtins import object
import logging
import traceback

from relaax.server.common.algorithm_loader import AlgorithmLoader
from .rlx_config import options

log = logging.getLogger(__name__)


class RLXWorker(object):

    @staticmethod
    def load_algorithm():
        try:
            options.Agent = AlgorithmLoader.load_agent(
                options.algorithm_path, options.algorithm_name)

        except Exception:
            log.critical("Can't load algorithm module")
            log.error(traceback.format_exc())
            exit()

    @staticmethod
    def load_protocol():
        try:
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
                raise

        except Exception:
            log.critical("Can't load protocol")
            exit()

    @classmethod
    def run(cls, socket, address):
        try:
            cls.load_protocol()
            cls.load_algorithm()
            log.debug('Running worker on connection %s:%d' % address)
            options.protocol.adoptConnection(socket, address)

        except Exception:
            log.error("Something crashed in the worker")
            log.error(traceback.format_exc())
