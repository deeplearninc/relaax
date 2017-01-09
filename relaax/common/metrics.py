from __future__ import print_function


class Metrics(object):
    def scalar(self, name, y, x=None):
        raise NotImplementedError
