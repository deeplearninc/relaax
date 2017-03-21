import time
import logging

# Load configuration options
# do it as early as possible
from parameter_server_config import options
from parameter_server_base import ParameterServerImpl
from relaax.server.common.bridge.bridge_server import BridgeServer

log = logging.getLogger(__name__)


class ParameterServer(object):

    @staticmethod
    def load_algorithm_ps():
        from relaax.server.common.algorithm_loader import AlgorithmLoader
        try:
            algorithm = AlgorithmLoader.load(options.algorithm_path)
        except Exception:
            log.critical("Can't load algorithm")
            raise

        use_system_ps = options.get("algorithm/use_system_ps", True)
        if not use_system_ps and hasattr(algorithm, "ParameterServer"):
            return algorithm.ParameterServer()
        elif use_system_ps and hasattr(algorithm, "TFGraph"):
            return ParameterServerImpl(algorithm.TFGraph())

        log.critical("Can't load algorithm's ParameterServer or TFGraph")
        raise

    @staticmethod
    def start():
        try:
            ps = ParameterServer.load_algorithm_ps()

            log.info("Staring parameter server server on %s:%d" % options.bind)

            # keep the server or else GC will stop it
            server = BridgeServer(options.bind, ps)
            server.start()

            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            # swallow KeyboardInterrupt
            pass
        except:
            raise


def main():
    ParameterServer.start()


if __name__ == '__main__':
    main()
