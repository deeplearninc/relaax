from __future__ import print_function

import imp
import os.path
import sys


def load(full_path):
    path, name = os.path.split(full_path)
    if path == '':
        path = '.'
    return _load_module(path, name)


def _load_module(path, name):
    if name not in sys.modules:
        file, pathname, description = imp.find_module(
            name,
            [os.path.expanduser(path)]
        )
        try:
            imp.load_module(name, file, pathname, description)
        finally:
            if file:
                file.close()
    return sys.modules[name]
