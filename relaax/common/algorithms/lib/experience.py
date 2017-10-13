from builtins import object


class Experience(object):
    def __init__(self, *args):
        self._lists = {k: [] for k in args}
        self._len = 0

    def __len__(self):
        return self._len

    def __getitem__(self, name):
        return self._lists[name]

    def push_record(self, **kwargs):
        for k, v in self._lists.items():
            v.append(kwargs[k])
        self._len += 1

    def del_record(self, size=1):
        for k, v in self._lists.items():
            del v[:size]
        self._len -= size
