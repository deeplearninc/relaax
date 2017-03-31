import functools


class SubgraphMeta(type):
    def __new__(meta, name, bases, dct):
        cls = super(SubgraphMeta, meta).__new__(meta, name, bases, dct)
        if hasattr(cls, 'assemble'):
            assemble = getattr(cls, 'assemble')
            factory = getattr(bases[0], '__factory__')
            setattr(cls, 'assemble', functools.partial(factory, cls, assemble))
        return cls


class Subgraph(object):
    __metaclass__ = SubgraphMeta

    @staticmethod
    def __factory__(target, method, *args, **kwargs):
        instance = target(method)
        instance._instance = instance
        return instance._build(*args, **kwargs)

    def __init__(self, func=None, instance=None):
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
