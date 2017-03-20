import traceback

import logging
log = logging.getLogger(__name__)

### Load configuration options 
### do it as early as possible 
from rlx_config import options
from rlx_port import RLXPort

class RLXServer():

    @staticmethod
    def preloadAlgorithm():
        try:
            from relaax.server.common.algorithm_loader import AlgorithmLoader
            options.algorithm_module = AlgorithmLoader.load(options.algorithm_path)

        except Exception as e:
            log.critical("Can't load algorithm")
            log.error(traceback.format_exc())            
            exit()

    @staticmethod
    def preloadProtocol():
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

        except Exception as e:
            log.critical("Can't load protocol")
            exit()

    @classmethod
    def start(cls):
        cls.preloadProtocol()
        cls.preloadAlgorithm()

        log.info("Starting RLX server on %s:%d"%options.bind)
        log.info("Expecting parameter server on %s:%d"%options.parameter_server)
        RLXPort.listen(options.bind)

def main():
    RLXServer.start()

if __name__ == '__main__':
    main()
