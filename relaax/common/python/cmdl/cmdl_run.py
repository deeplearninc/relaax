from builtins import object
import os
import sys
import logging
from honcho.manager import Manager

from relaax.common.python.cmdl.cmdl_config import options

log = logging.getLogger("relaax")


class RManager(Manager):

    def __init__(self, *args, **kwargs):
        super(RManager, self).__init__(*args, **kwargs)

    def _any_stopped(self):
        clients = []
        for _, p in self._processes.items():
            if p['process'].name.startswith('client'):
                clients.append(p.get('returncode') is not None)
        if len(clients):
            return all(clients)
        else:
            super(RManager, self)._any_stopped()


class CmdlRun(object):

    def __init__(self):
         if sys.platform == 'win32':
            self.nobuffer = ''
        else:
            self.nobuffer = 'PYTHONUNBUFFERED=true'
    
        self.cmdl = "--config %s " % options.config
        if options.cmdl_log_level:
            self.cmdl += "--log-level %s" % options.cmdl_log_level

    def run_components(self):

        manager = RManager()

        self.run_parameter_server(manager)
        self.run_rlx_server(manager)
        self.run_client(manager)

        manager.loop()
        sys.exit(manager.returncode)

    def run_parameter_server(self, manager):
        if options.run in ['all', 'servers', 'parameter-server']:
            manager.add_process('parameter-server',
                                '%s relaax-parameter-server %s'
                                % (self.nobuffer, self.cmdl))

    def run_rlx_server(self, manager):
        if options.run in ['all', 'servers', 'rlx-server']:
            manager.add_process('rlx-server',
                                '%s relaax-rlx-server %s'
                                % (self.nobuffer, self.cmdl))

    def run_client(self, manager):
        if options.run in ['all', 'client']:
            if options.client:
                self.run_all_clients(manager)
            else:
                color_seq = "\033[1;31m"
                reset_seq = "\033[0m"
                log.info(color_seq + "No client specified" + reset_seq)

    def run_all_clients(self, manager):
        if options.concurrent > 1:
            count = 0
            while count < options.concurrent:
                self.run_one_client('client-%d' % count, manager)
                count += 1
        else:
            self.run_one_client('client', manager)

    def run_one_client(self, process_name, manager):
        manager.add_process(
            process_name, '%s python %s %s' % (self.nobuffer, options.client, self.cmdl))


def main():
    # Disable TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # run
    CmdlRun().run_components()


if __name__ == '__main__':
    main()
