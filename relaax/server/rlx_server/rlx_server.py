from __future__ import absolute_import
import glob
import logging
import os

# Load configuration options
# do it as early as possible
from .rlx_port import RLXPort
from .rlx_config import options

log = logging.getLogger(__name__)


class RLXServer():

    @classmethod
    def start(cls):
        if hasattr(options.relaax_rlx_server, 'profile_dir'):
            for f in glob.glob(os.path.join(
                    options.relaax_rlx_server.profile_dir, 'rlx_*.txt')):
                os.remove(f)
        log.info("Starting RLX server on %s:%d" % options.bind)
        log.info("Expecting parameter server on %s:%d" % options.parameter_server)
        log.info("Expecting metrics server on %s:%d" % options.metrics_server)
        RLXPort.listen(options.bind)


def main():
    RLXServer.start()


if __name__ == '__main__':
    main()
