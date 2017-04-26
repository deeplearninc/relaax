import logging
import time
import traceback

# Load configuration options
# do it as early as possible
from rlx_port import RLXPort
from rlx_config import options
from relaax.server.common.algorithm_loader import AlgorithmLoader

log = logging.getLogger(__name__)


class RLXServer():

    @staticmethod
    def preload_algorithm():
        try:
            options.algorithm_module = AlgorithmLoader.load(options.algorithm_path)

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
                import rlx_protocol.rawsocket.rlx_protocol as protocol
                options.protocol = protocol
            elif name == 'twisted':
                log.debug("Loading protocol on twisted")
                import rlx_protocol.twisted.rlx_protocol as protocol
                options.protocol = protocol
            else:
                raise

        except Exception:
            log.critical("Can't load protocol")
            exit()

    @classmethod
    def start(cls):
        cls.preload_protocol()
        cls.preload_algorithm()
        log.info("Starting RLX server on %s:%d" % options.bind)
        log.info("Expecting parameter server on %s:%d" % options.parameter_server)
        RLXPort.listen(options.bind)


def main():
    time.sleep(15)
    RLXServer.start()


if __name__ == '__main__':
    main()
