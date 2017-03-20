import traceback

import logging
log = logging.getLogger(__name__)

from rlx_config import options

class RLXWorker():

    @classmethod
    def run(cls,socket,address):
        try:
            log.debug('Running worker on connection %s:%d'%address)
            options.protocol.adoptConnection(socket,address)

        except Exception as error:
            log.error("Something crashed in the worker")
            log.error(traceback.format_exc())
  
