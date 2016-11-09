
class C(object):
    def __init__(self, v):
        self._v = v

    def m(self):
        return self._v

print C('v').m
print (C('v').m)()

def f(a, b):
    pass

l = lambda *args: f(*args)

l(1, 2, 3)


