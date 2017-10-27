from builtins import object
import os
import sys
import click
import logging
log = logging.getLogger("relaax")


CONTEXT_SETTINGS = dict(auto_envvar_prefix='RELAAX')
ALGORITHMS = ['policy-gradient', 'da3c', 'trpo', 'ddpg', 'dqn', 'dppo']
ALGORITHMS_META = '|'.join(ALGORITHMS)
ENVIRONMENTS = ['basic', 'openai-gym', 'deepmind-lab']
ENVIRONMENTS_META = '|'.join(ENVIRONMENTS)


class Context(object):

    def __init__(self):
        pass

    def log(self, msg, *args, **kwargs):
        log.info(msg, *args, **kwargs)

    @staticmethod
    def setup_logger(format='%(asctime)s %(name)s | %(message)s'):
        logging.basicConfig(
            stream=sys.stdout,
            datefmt='%H:%M:%S',
            format=format,
            level=logging.INFO)


pass_context = click.make_pass_decorator(Context, ensure=True)


class RelaaxCLI(click.MultiCommand):
    cmd_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), 'commands'))

    def list_commands(self, ctx):
        rv = []
        for filename in os.listdir(RelaaxCLI.cmd_folder):
            if filename.endswith('.py') and \
               filename.startswith('cmd_'):
                rv.append(filename[4:-3])
        rv.sort()
        return rv

    def get_command(self, ctx, name):
        try:
            if sys.version_info[0] == 2:
                name = name.encode('ascii', 'replace')
            mod = __import__('relaax.cmdl.commands.cmd_' + name,
                             None, None, ['cli'])
        except ImportError as e:
            print(name)
            print(str(e))
            return
        return mod.cmdl


@click.command(cls=RelaaxCLI, context_settings=CONTEXT_SETTINGS)
@pass_context
def cmdl(ctx):
    """RELAAX command line interface."""
