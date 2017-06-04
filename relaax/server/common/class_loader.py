from builtins import object

import imp
import importlib
import os
import sys


class ClassLoader(object):

    @staticmethod
    def load(path, cname):
        if path is None:
            pname, cname = cname.rsplit('.', 1)
        else:
            mpath, mname = os.path.split(path)

            if os.path.isdir(path):
                pname, cname = cname.rsplit('.', 1)
            elif os.path.isfile(path):
                mname, ext = os.path.splitext(mname)
                pname = None
            else:
                raise ImportError("No module named %s" % path)

            module = ClassLoader.load_module(mname, mpath)

        if pname:
            module = importlib.import_module(pname)

        return getattr(module, cname)

    @staticmethod
    def load_module(mname, mpath):
        if mname in sys.modules:
            return sys.modules[mname]
        file, pathname, description = imp.find_module(mname, [mpath])
        try:
            return imp.load_module(mname, file, pathname, description)
        finally:
            if file:
                file.close()
