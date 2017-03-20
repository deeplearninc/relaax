import imp
import sys
import os.path

class AlgorithmLoader():
    
    @classmethod
    def load(cls,full_path):
        path, name = os.path.split(full_path)
        if path == '':
            path = '.'
        return cls._load_module(path, name)

    @classmethod
    def _load_module(cls,path, name):
        if name not in sys.modules:
            file, pathname, description = imp.find_module(name, [path])
            try:
                imp.load_module(name, file, pathname, description)
            finally:
                if file:
                    file.close()
        return sys.modules[name]
