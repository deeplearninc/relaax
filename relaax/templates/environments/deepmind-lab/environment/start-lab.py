from __future__ import print_function

import os
import shutil
import os.path
import logging
import subprocess

from relaax.environment.config import options

log = logging.getLogger(__name__)


class MDLab(object):

    def __init__(self):
        self.show_ui = options.get('show_ui', False)
        self.lab_path = options.get('environment/lab_path', '/lab')
        self.level_script = options.get('environment/level_script', None)
        if self.level_script is None:
            self.level_script = ''
        else:
            self.level_script = ' --level_script %s' % self.level_script

    def start(self):
        if self.show_ui:
            subprocess.call("/opt/startup.sh", shell=True)
        self._set_entry_point()
        if len(self.level_script) > 0:
            self._copy_maps()
        self._run_deepmind_lab()

    def _set_entry_point(self):
        log.info('Copy entrypoint to random_agent')
        random_agent = os.path.join(self.lab_path, 'python/random_agent.py')
        old_random_agent = random_agent + '.old'
        module_path = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(old_random_agent):
            shutil.copy2(random_agent, old_random_agent)
        shutil.copy2(os.path.join(module_path, 'entrypoint.py'), random_agent)

    def _copy_maps(self):
        game_scripts = os.path.join(self.lab_path, 'assets/game_scripts/custom-map/')
        log.info('Copy environment/custom-map/ to %s' % game_scripts)
        if os.path.exists(game_scripts):
            shutil.rmtree(game_scripts)
        shutil.copytree('environment/custom-map/', game_scripts)

    def _run_deepmind_lab(self):
        log.info('Run deepmind-lab, please wait, it may take a moment...')
        try:
            rlx_address = options.get('rlx_server_address', None)
            if rlx_address is None:
                rlx_address = options.get('relaax_rlx_server/bind', 'localhost:7001')
            app_path = os.path.dirname(os.path.abspath(__file__))
            config = os.path.abspath(os.path.join(app_path, '../app.yaml'))
            headless = 'false' if self.show_ui else 'osmesa'
            cmd = 'cd %s && bazel run :random_agent --define headless=%s' % (self.lab_path, headless)
            cmd = '%s --%s --app_path %s --config %s --show-ui %s --rlx-server-address %s' % \
                (cmd, self.level_script, app_path, config, self.show_ui, rlx_address)
            log.info(cmd)
            subprocess.call(cmd, shell=True)
        except subprocess.CalledProcessError as e:
            log.info('Error while building deepmind-lab: %s' % str(e))
            raise


def main():
    MDLab().start()


if __name__ == '__main__':
    main()
