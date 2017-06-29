from builtins import object
import io


class MockSocket(object):

    def __init__(self, sin, sout):
        self.output = sin
        self.input = sout
        self.opened = True

    def bind(self, address):
        self.address = address

    def listen(self, *args):
        pass

    def accept(self, *args):
        return self.create(), self.address

    def setsockopt(self, *args):
        pass

    def connect(self, *args):
        pass

    def sendall(self, data):
        assert self.opened
        self.output.sendall(data)

    def recv(self, n):
        assert self.opened
        return self.input.recv(n)

    def shutdown(self, how):
        assert self.opened
        return True
    
    def close(self):
        assert self.opened
        self.opened = False

    @staticmethod
    def create():
        sktbuf = MockSocketBuffer()
        return MockSocket(sktbuf, sktbuf)


class MockSocketBuffer(object):
    def __init__(self):
        self._buffer = io.BytesIO()
        self._read_pos = self._buffer.tell()

    def sendall(self, data):
        self._buffer.seek(0, io.SEEK_END)
        self._buffer.write(data)

    def recv(self, n):
        self._buffer.seek(self._read_pos, io.SEEK_SET)
        bs = self._buffer.read(n)
        self._read_pos = self._buffer.tell()
        return bs
