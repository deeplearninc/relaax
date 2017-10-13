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
    add('--level_script', type=str, default=None, required=False, help='Custom level maps')
    args = parser.parse_args()
    if args.app_path:
        if '--app_path' in sys.argv:
            index = sys.argv.index('--app_path')
            del sys.argv[index]
            del sys.argv[index]
        if '--level_script' in sys.argv:
            index = sys.argv.index('--level_script')
            del sys.argv[index]
            del sys.argv[index]
        _, app_module = os.path.split(args.app_path)
        training = ClassLoader.load(args.app_path, '%s.%s' % (app_module, 'training.Training'))
        training().run()
    else:
        print('Please provide application path to load...')
