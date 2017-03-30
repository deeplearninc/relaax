from relaax.common.python.config.loaded_config import options

# sub-graph
class SubGraph(object):
    @classmethod
    def assemble(cls, *args, **kwargs):
        return SubGraphPointer(cls().build(*args, **kwargs))


class SubGraphPointer(object):
    def __init__(self, pointer):
        self.pointer = pointer

    @property
    def tensor(self):
        return self.pointer

    @property
    def op(self):
        return self.pointer
