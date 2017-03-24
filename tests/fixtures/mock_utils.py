

class MockUtils(object):

    @staticmethod
    def raise_(exception):
        raise exception

    @staticmethod
    def called_with(target, method, monkeypatch):
        def method_mock(*args):
            called_with[0] = args[1]

        called_with = [None]
        monkeypatch.setattr(target, method, method_mock)
        return called_with

    @staticmethod
    def called_once(target, method, monkeypatch):
        def method_mock(*args):
            called_times[0] += 1

        called_times = [0]
        monkeypatch.setattr(target, method, method_mock)
        return called_times
