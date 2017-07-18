from builtins import object

import imp
import importlib
import os
import sys


class ClassLoader(object):

    @classmethod
    def load(cls, path, cname):
        module = cls.import_module(path, cname)
        if path is None or os.path.isdir(path):
            _, cname = cname.rsplit('.', 1)
        return getattr(module, cname)

    @classmethod
    def import_module(cls, path, cname):
        if path is None:
            pname, _ = cname.rsplit('.', 1)
        else:
            mpath, mname = os.path.split(path)

            if os.path.isdir(path):
                pname, _ = cname.rsplit('.', 1)
            elif os.path.isfile(path):
                mname, _ = os.path.splitext(mname)
                pname = None
            else:
                raise ImportError("No module named %s" % path)

            module = cls.load_module(mname, mpath)

        if pname:
            module = importlib.import_module(pname)

        return module

    @classmethod
    def load_module(cls, mname, mpath):
        if mname in sys.modules:
            return sys.modules[mname]
        file, pathname, description = imp.find_module(mname, [mpath])
        try:
            return imp.load_module(mname, file, pathname, description)
        finally:
            if file:
                file.close()
