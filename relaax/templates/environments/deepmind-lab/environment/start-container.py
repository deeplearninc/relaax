from __future__ import print_function
import sys
import uuid
import signal
import logging
import platform
import subprocess

from relaax.environment.config import options

log = logging.getLogger(__name__)

CONTAINER_NAME = str(uuid.uuid1())


def signal_handler(signal, frame):
    log.info('stopping %s' % CONTAINER_NAME)
    subprocess.call('docker stop %s' % CONTAINER_NAME, shell=True)
    sys.exit(0)
signal.signal(signal.SIGTERM, signal_handler)


class MDLabContainer(object):

    def __init__(self, image_name):
        self.image_name = image_name
        self.show_ui = options.get('show_ui', False)

    def start(self):
        if not self._is_container_exists():
            log.info("Building container")
            self._build_container()
        if platform.system() == 'Darwin' and self.show_ui:
            log.info('Starting VNC viewer...')
            cmd = 'sleep 5 && open vnc://user:password@127.0.0.1:5901'
            log.info(cmd)
            subprocess.Popen(cmd, shell=True)
        log.info("Starting docker container...")
        self._start_container()

    @staticmethod
    def _get_rlx_address():
        def parse_address(address):
            try:
                host, port = address.split(':')
                return host, int(port)
            except Exception:
                raise ValueError("Can't parse RLX server address.")

        host, port = parse_address(options.get('relaax_rlx_server/bind', '0.0.0.0:7001'))
        if platform.system() == 'Linux':
            host = '127.0.0.1'
        elif host == '0.0.0.0':
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            host = s.getsockname()[0]
            s.close()

        return '%s:%d' % (host, port)

    def _start_container(self):
        try:
            net = ''
            if platform.system() == 'Linux':
                net = '--net host'
            rlx_address = self._get_rlx_address()
            cmd = 'docker run %s --name %s -t %s -v ${PWD}:/app %s --show-ui %s --rlx-server-address %s'
            cmd = cmd % (net, CONTAINER_NAME, '-p 5901:5901' if self.show_ui else '',
                         self.image_name, self.show_ui, rlx_address)
            log.info(cmd)
            subprocess.call(cmd, shell=True)
        except subprocess.CalledProcessError as e:
            log.info('Start container error: %s' % str(e))
            raise

    def _build_container(self):
        try:
            cmd = ('docker build -f ./environment/.docker/Dockerfile -t '
                   '%s ./environment/.docker/' % self.image_name)
            log.info(cmd)
            subprocess.call(cmd, shell=True)
        except subprocess.CalledProcessError as e:
            log.info('Build container error: %s' % str(e))
            raise

    def _is_container_exists(self):
        try:
            subprocess.check_output('docker inspect --type=image %s' % self.image_name, shell=True)
        except subprocess.CalledProcessError:
            return False
        return True

    def _is_container_running(self):
        s = subprocess.check_output('docker ps', shell=True)
        return s.find(self.image_name) != -1


def main():
    MDLabContainer('deepmind-lab:1.0.0').start()


if __name__ == '__main__':
    main()
