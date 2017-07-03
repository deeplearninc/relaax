from __future__ import print_function

from builtins import object

import sys
import ruamel.yaml


class CommentedYaml(object):

    @staticmethod
    def write(ctx, filename, config):
        try:
            with open(filename, 'w') as out:
                out.write(ruamel.yaml.dump(config, Dumper=ruamel.yaml.RoundTripDumper))
        except EnvironmentError as e:   # parent of IOError, OSError *and* WindowsError where available
            ctx.log('Sorry, can\'t write config: %s' % str(e))
            sys.exit()

    @staticmethod
    def read(ctx, filename, error_message):
        try:
            with open(filename, 'r') as inp:
                yaml = ruamel.yaml.load(inp, ruamel.yaml.RoundTripLoader)
        except EnvironmentError as e:   # parent of IOError, OSError *and* WindowsError where available
            ctx.log(error_message % str(e))
            sys.exit()
        return yaml
