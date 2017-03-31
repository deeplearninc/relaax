import functools


class SubgraphMeta(type):
    def __new__(meta, name, bases, dct):
        cls = super(SubgraphMeta, meta).__new__(meta, name, bases, dct)
        if hasattr(cls, 'assemble'):
            assemble = getattr(cls, 'assemble')
            setattr(cls, 'assemble', functools.partial(meta.factory, cls, assemble))
            setattr(cls, '__former_assemble__', assemble)
        return cls

    @staticmethod
    def factory(target, method, *args, **kwargs):
        instance = target(method)
        instance._instance = instance
        return instance._build(*args, **kwargs)


class Subgraph(object):
    __metaclass__ = SubgraphMeta

    def __init__(self, func=None, instance=None):
        if func is None:
            func = self.__former_assemble__
        self._func = func
        self._instance = instance
        setattr(self, 'assemble', self._build)
        self._assembled = False
        self._ptr = None

    def __call__(self, *args, **kwargs):
        return self._pointer()

    def _build(self, *args, **kwargs):
        if not self._assembled:
            self._assembled = True
            if self._instance is None:
                self._ptr = self._func(*args, **kwargs)
            else:
                self._ptr = self._func(self._instance, *args, **kwargs)
        return self

    def _pointer(self):
        if not self._assembled:
            raise Exception('subgraph is not built yet'
                            'please use assemble to build sub-graph')
        return self._ptr

    @property
    def tensor(self):
        return self._pointer()

    @property
    def op(self):
        return self._pointer()
