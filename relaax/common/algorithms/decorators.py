from subgraph import Subgraph

OPS_CACHE_NAME = '__subgraph_operations__'


# decorator
class define_subgraph(Subgraph):

    def __init__(self, func):
        super(define_subgraph, self).__init__(func)

    def __get__(self, instance, cls):
        if self._instance is None:
            self._instance = instance
            self.add_to_ops_cache()
        return self

    def add_to_ops_cache(self):
        if not hasattr(self._instance, OPS_CACHE_NAME):
            setattr(self._instance, OPS_CACHE_NAME, {})
        cache = getattr(self._instance, OPS_CACHE_NAME)
        cache[self._func.__name__] = self
