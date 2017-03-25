from collections import namedtuple


class MockUtils(object):

    @staticmethod
    def raise_(exception):
        raise exception

    @staticmethod
    def called_with(target, method, monkeypatch):
        def method_mock(*args, **kwargs):
            called_with.args = args
            called_with.kwargs = kwargs

        called_with = namedtuple('CalledWith', 'args kwargs')
        monkeypatch.setattr(target, method, method_mock)
        return called_with

    @staticmethod
    def called_once(target, method, monkeypatch):
        def method_mock(*args):
            called_times[0] += 1

        called_times = [0]
        monkeypatch.setattr(target, method, method_mock)
        return called_times
