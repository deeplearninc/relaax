from __future__ import absolute_import
import logging
import traceback

# Load configuration options
# do it as early as possible
from .rlx_port import RLXPort
from .rlx_config import options

log = logging.getLogger(__name__)


class RLXServer():

    @classmethod
    def start(cls):
        log.info("Starting RLX server on %s:%d" % options.bind)
        log.info("Expecting parameter server on %s:%d" % options.parameter_server)
        RLXPort.listen(options.bind)


def main():
    RLXServer.start()


if __name__ == '__main__':
    main()
