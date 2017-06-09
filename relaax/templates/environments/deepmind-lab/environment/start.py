import os
import sys
import uuid
import shutil
import signal
import os.path
import platform
import subprocess

from config import options

import logging
log = logging.getLogger(__name__)

CONTAINER_NAME = str(uuid.uuid1())


def signal_handler(signal, frame):
    print 'stopping %s' % CONTAINER_NAME
    subprocess.call('docker stop %s' % CONTAINER_NAME, shell=True)
    sys.exit(0)
signal.signal(signal.SIGTERM, signal_handler)


class MDLab(object):

    def __init__(self, image_name):
        self.image_name = image_name
        self.show_ui = options.get('show_ui', False)
        self.lab_path = options.get('environment/lab_path', '/lab')

    def start(self):
        if platform.system() == 'Linux':
            if self.show_ui:
                subprocess.call("/opt/startup.sh", shell=True)
            self._set_entry_point()
            self._run_deepmind_lab()
        else:
            print 'Running not on Linux, have to start in docker'
            if not self._is_container_exists():
                print "Building container"
                self._build_container()
            if platform.system() == 'Darwin' and self.show_ui:
                print 'Starting VNC viewer...'
                cmd = 'sleep 5 && open vnc://user:password@127.0.0.1:5901'
                print cmd
                subprocess.Popen(cmd, shell=True)
            print "Starting docker container"
            self._start_container()

    def _set_entry_point(self):
        print 'Copy entrypoint to random_agent'
        random_agent = os.path.join(self.lab_path, 'python/random_agent.py')
        old_random_agent = random_agent + '.old'
        module_path = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(old_random_agent):
            shutil.copy2(random_agent, old_random_agent)
        shutil.copy2(os.path.join(module_path, 'entrypoint.py'), random_agent)

    def _run_deepmind_lab(self):
        print 'Run deepmind-lab, please wait, it may take a moment...'
        try:
            rlx_address = options.get('rlx_server_address', None)
            if rlx_address is None:
                rlx_address = options.get('relaax_rlx_server/bind', 'localhost:7001')
            app_path = os.path.dirname(os.path.abspath(__file__))
            config = os.path.abspath(os.path.join(app_path, '../app.yaml'))
            headless = 'false' if self.show_ui else 'osmesa'
            cmd = 'cd %s && bazel run :random_agent --define headless=%s' % (self.lab_path, headless)
            cmd = '%s -- --app_path %s --config %s --show-ui %s --rlx-server-address %s' % \
                (cmd, app_path, config, self.show_ui, rlx_address)
            print cmd
            subprocess.call(cmd, shell=True)
        except subprocess.CalledProcessError as e:
            print 'Error while building deepmind-lab: %s' % str(e)
            raise

    def _get_rlx_address(self):
        def parse_address(address):
            try:
                host, port = address.split(':')
                return host, int(port)
            except Exception:
                raise ValueError("Can't parse RLX server address.")

        host, port = parse_address(options.get('relaax_rlx_server/bind', '0.0.0.0:7001'))
        if host == '0.0.0.0':
            host = subprocess.check_output(
                'ifconfig | grep -E "([0-9]{1,3}\.){3}[0-9]{1,3}" | grep -v 127.0.0.1 |'
                ' awk \'{ print $2 }\' | cut -f2 -d: | head -n1',
                shell=True).strip()
        return '%s:%d' % (host, port)

    def _start_container(self):
        try:
            rlx_address = self._get_rlx_address()
            cmd = ('docker run --name %s -t -p 5901:5901 -v ${PWD}:/app'
                   ' %s --show-ui %s --rlx-server-address %s')
            cmd = cmd % (CONTAINER_NAME, self.image_name, self.show_ui, rlx_address)
            print cmd
            subprocess.call(cmd, shell=True)
        except subprocess.CalledProcessError as e:
            print 'Start container error: %s' % str(e)
            raise

    def _build_container(self):
        try:
            cmd = 'docker build -f ./environment/.docker/Dockerfile -t %s ./environment/.docker/' % self.image_name
            print cmd
            subprocess.call(cmd, shell=True)
        except subprocess.CalledProcessError as e:
            print 'Build container error: %s' % str(e)
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
    MDLab('deepmind-lab:1.0.0').start()


if __name__ == '__main__':
    main()
