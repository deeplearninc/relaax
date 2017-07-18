from builtins import object


class Metrics(object):
    def scalar(self, name, y, x=None):
        raise NotImplementedError

    def histogram(self, name, y, x=None):
        raise NotImplementedError

    def update(self, metrics):
        for metric in metrics:
            getattr(self, metric['method'])(name=metric['name'], y=metric['y'], x=metric['x'])
