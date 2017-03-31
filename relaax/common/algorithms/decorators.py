# sub-graph
class SubGraph(object):

    def __init__(self, obj=None, func=None):
        self._obj = obj
        self._func = self.build if func is None else func
        self._assembled = False
        self._pointer = None

    def __call__(self, *args, **kwargs):
        return self.__pointer()

    def assemble(self, *args, **kwargs):
        if not self._assembled:
            self._assembled = True
            if self._obj is None:
                self._pointer = self._func(*args, **kwargs)
            else:
                self._pointer = self._func(self._obj, *args, **kwargs)
        return self

    @property
    def tensor(self):
        return self.__pointer()

    @property
    def op(self):
        return self.__pointer()

    def __pointer(self):
        if not self._assembled:
            print 'please use assemble to build sub-graph'
        return self._pointer


# define_subgraph decorator
OPS_CACHE_NAME = '__subgraph_operations__'


class define_subgraph(object):

    def __init__(self, func):
        self._func = func
        self._attribute = '_cache_' + func.__name__

    def __get__(self, instance, cls):
        if not hasattr(instance, self._attribute):
            subgraph = SubGraph(instance, self._func)
            self.add_to_ops(instance, subgraph)
            setattr(instance, self._attribute, subgraph)
        return getattr(instance, self._attribute)

    def add_to_ops(self, instance, subgraph):
        if not hasattr(instance, OPS_CACHE_NAME):
            setattr(instance, OPS_CACHE_NAME, {})
        cache = getattr(instance, OPS_CACHE_NAME)
        cache[subgraph._func.__name__] = subgraph
