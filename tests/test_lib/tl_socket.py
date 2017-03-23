import io


class TLSocket(object):
    def __init__(self, sin, sout):
        self.output = sin
        self.input = sout
        self.opened = True

    def sendall(self, data):
        assert self.opened
        self.output.sendall(data)

    def recv(self, n):
        assert self.opened
        return self.input.recv(n)

    def close(self):
        assert self.opened
        self.opened = False


class TLSocketBuffer(object):
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
