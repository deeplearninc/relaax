
class Subgraph(object):
    def __init__(self, *args, **kwargs):
        self.__pointer = self.build(*args, **kwargs)

    @property
    def tensor(self):
        return self.__pointer

    @property
    def op(self):
        return self.__pointer
