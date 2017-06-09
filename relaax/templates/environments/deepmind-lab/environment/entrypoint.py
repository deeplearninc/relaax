import sys
import os.path
import argparse
from relaax.server.common.class_loader import ClassLoader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add = parser.add_argument
    add('--app_path', type=str, help='The application to load')
    add('--config', type=str, help='The application config')
    add('--show-ui', type=str, help='The application config')
    add('--rlx-server-address', type=str, default=None, required=False, help='RELAAX RLX Server Address')
    args = parser.parse_args()
    if args.app_path:
        app_path_index = sys.argv.index('--app_path')
        del sys.argv[app_path_index]
        del sys.argv[app_path_index]
        _, app_module = os.path.split(args.app_path)
        training = ClassLoader.load(args.app_path, '%s.%s' % (app_module, 'training.Training'))
        training().run()
    else:
        print('Please provide application path to load...')
