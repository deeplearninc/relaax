from __future__ import print_function

import os
import sys
import glob
import click
import shutil
import string
import ruamel.yaml
from ..cmdl import pass_context


class CmdConfig(object):

    def __init__(self, ctx, config, algorithm):
        self.ctx = ctx
        self.config = config
        self.algorithm = algorithm

    def apply(self):

        app_config = self._read_yaml(self.config, 'Sorry, can\'t read input yaml: %s')

        module_path = os.path.dirname(os.path.abspath(__file__))
        algo_config_path = os.path.abspath(os.path.join(
            module_path, '../../templates/configs/%s.yaml' % self.algorithm))

        algo_config = self._read_yaml(algo_config_path, 'Sorry, can\'t read algorithm config: %s')

        app_config['algorithm'] = algo_config['algorithm']

        self._make_backup()

        self._write_yaml(self.config, app_config)

        self.ctx.log('Configured %s with %s algorithm default parameters.' %
                     (os.path.basename(self.config), self.algorithm))

    def _write_yaml(self, filename, config):
        try:
            with open(filename, 'w') as out:
                out.write(ruamel.yaml.dump(config, Dumper=ruamel.yaml.RoundTripDumper))
        except EnvironmentError as e:   # parent of IOError, OSError *and* WindowsError where available
            self.ctx.log('Sorry, can\'t write config: %s' % str(e))
            sys.exit()

    def _read_yaml(self, filename, error_message):
        try:
            with open(filename, 'r') as inp:
                yaml = ruamel.yaml.load(inp, ruamel.yaml.RoundTripLoader)
        except EnvironmentError as e:   # parent of IOError, OSError *and* WindowsError where available
            self.ctx.log(error_message % str(e))
            sys.exit()
        return yaml

    def _make_backup(self):
        """Save a numbered backup of config file."""
        # If the file doesn't already exist, there's nothing to do.
        if os.path.isfile(self.config):
            new_name = self._versioned_name(self._current_revision() + 1)
            shutil.copy2(self.config, new_name)

    def _versioned_name(self, revision):
        """Get config with a revision number appended."""
        return "%s.~%s~" % (self.config, revision)

    def _current_revision(self):
        """Get the revision number of config largest existing backup."""
        revisions = [0] + self._revisions()
        return max(revisions)

    def _revisions(self):
        """Get the revision numbers of all of config backups."""
        revisions = []
        backup_names = glob.glob("%s.~[0-9]*~" % (self.config))
        for name in backup_names:
            try:
                revision = int(string.split(name, "~")[-2])
                revisions.append(revision)
            except ValueError:
                # Some ~[0-9]*~ extensions may not be wholly numeric.
                pass
        revisions.sort()
        return revisions


@click.command('config', short_help='Configure algorithm.')
@click.option('--config', '-c', type=click.File(lazy=True), show_default=True, default='app.yaml',
              help='Relaax configuraion yaml file.')
@click.option('--algorithm', '-a', show_default=True, default='policy-gradient', metavar='',
              type=click.Choice(['policy-gradient', 'da3c', 'trpo']),
              help='[policy-gradient|da3c|trpo]\nRelaax algorithm name.')
@pass_context
def cmdl(ctx, config, algorithm):
    """Configure Relaax algorithm.

    Fill algorithm section of the configuration file with default parameters for the specified algorithm.
    Old configuration file will be copied to backup file.
    """
    ctx.setup_logger(format='')
    CmdConfig(ctx, config.name, algorithm).apply()
