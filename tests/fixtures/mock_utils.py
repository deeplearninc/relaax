

class MockUtils(object):

    class Placeholder(object):
        def __init__(self):
            self.args = None
            self.kwargs = None
            self.times = 0

    @staticmethod
    def raise_(exception):
        raise exception

    @staticmethod
    def called_with(target, method, monkeypatch):
        def method_mock(*args, **kwargs):
            called_with.args = args
            called_with.kwargs = kwargs

        called_with = MockUtils.Placeholder()
        monkeypatch.setattr(target, method, method_mock)
        return called_with

    @staticmethod
    def count_calls(target, method, monkeypatch):
        def method_mock(*args):
            called.times += 1

        called = MockUtils.Placeholder()
        monkeypatch.setattr(target, method, method_mock)
        return called
