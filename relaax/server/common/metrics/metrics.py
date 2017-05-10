from builtins import object
class Metrics(object):
    def scalar(self, name, y, x=None):
        raise NotImplementedError
