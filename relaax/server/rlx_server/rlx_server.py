from __future__ import absolute_import
import glob
import logging
import os
import traceback

# Load configuration options
# do it as early as possible
from .rlx_port import RLXPort
from .rlx_config import options
from relaax.server.common.algorithm_loader import AlgorithmLoader

log = logging.getLogger(__name__)


class RLXServer():

    @staticmethod
    def preload_algorithm():
        try:
            options.Agent = AlgorithmLoader.load_agent(
                    options.algorithm_path, options.algorithm_name)

        except Exception:
            log.critical("Can't load algorithm module")
            log.error(traceback.format_exc())
            exit()

    @staticmethod
    def preload_protocol():
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
    def start(cls):
        if hasattr(options.relaax_rlx_server, 'profile_dir'):
            for f in glob.glob(os.path.join(
                    options.relaax_rlx_server.profile_dir, 'rlx_*.txt')):
                os.remove(f)
        cls.preload_protocol()
        cls.preload_algorithm()
        log.info("Starting RLX server on %s:%d" % options.bind)
        log.info("Expecting parameter server on %s:%d" % options.parameter_server)
        RLXPort.listen(options.bind)


def main():
    RLXServer.start()


if __name__ == '__main__':
    main()
