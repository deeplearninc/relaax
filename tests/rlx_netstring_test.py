from mock import Mock
from test_lib.tl_socket import TLSocket, TLSocketBuffer

from relaax.common.rlx_netstring import NetString, NetStringClosed, NetStringException


class TestNetString:

    def test_to_wire_and_read_back(self):
        sktbuf = TLSocketBuffer()
        socket = TLSocket(sktbuf, sktbuf)
        some_string = "some string"
        nstr = NetString(socket)
        nstr.write_string(some_string)
        data = nstr.read_string()
        assert data == some_string

    def test_recieve_zero_length(self):
        sktbuf = TLSocketBuffer()
        socket = TLSocket(sktbuf, sktbuf)
        nstr = NetString(socket)
        # close on empty socket
        try:
            nstr.read_string()
            assert False
        except NetStringClosed as e:
            assert str(e) == 'connection closed'

    def test_not_numbers_in_string_length(self):
        # close on not numbers in string length
        sktbuf = TLSocketBuffer()
        socket = TLSocket(sktbuf, sktbuf)
        sktbuf.sendall('abc')
        nstr = NetString(socket)
        try:
            nstr.read_string()
            assert False
        except NetStringException as e:
            assert str(e) == 'can\'t receive, wrong net string format'

    def test_too_long_string(self):
        sktbuf = TLSocketBuffer()
        socket = TLSocket(sktbuf, sktbuf)
        data = str(NetString.MAX_STRING_LEN)
        sktbuf.sendall(data)
        nstr = NetString(socket)
        try:
            data = nstr.read_string()
            assert False
        except NetStringException as e:
            assert str(e) == 'can\'t receive, wrong net string format'

    def test_slen_greater_max(self):
        sktbuf = TLSocketBuffer()
        socket = TLSocket(sktbuf, sktbuf)
        data = str(NetString.MAX_STRING_LEN - 1) + ":abc"
        NetString.MAX_STRING_LEN -= 2
        sktbuf.sendall(data)
        nstr = NetString(socket)
        try:
            data = nstr.read_string()
            assert False
        except NetStringException as e:
            assert str(e) == 'net string too long'

    def test_comma_at_end_of_string(self):
        sktbuf = TLSocketBuffer()
        socket = TLSocket(sktbuf, sktbuf)
        data = "some string"
        data = '%d:%sx' % (len(data), data)
        sktbuf.sendall(data)
        nstr = NetString(socket)
        try:
            data = nstr.read_string()
            assert False
        except NetStringException as e:
            assert str(e) == 'wrong net string format'

    def test_too_short_string(self):
        sktbuf = TLSocketBuffer()
        socket = TLSocket(sktbuf, sktbuf)
        data = "some string"
        data = '%d:%s' % (len(data), data)
        sktbuf.sendall(data)
        nstr = NetString(socket)
        try:
            data = nstr.read_string()
            assert False
        except NetStringClosed as e:
            assert str(e) == 'can\'t receive, net string closed'

    def test_writing_to_long_string(self):
        nstr = NetString(None)
        data = Mock()
        data.__len__ = Mock()
        data.__len__.return_value = NetString.MAX_STRING_LEN + 1
        try:
            data = nstr.write_string(data)
            assert False
        except NetStringException as e:
            assert str(e) == 'can\'t send, net string too long'

    def test_exception_in_send_all(self):
        socket = Mock()
        socket.sendall = Mock()
        socket.sendall = Mock(side_effect=Exception('some error'))
        nstr = NetString(socket)
        try:
            nstr.write_string("some data")
            assert False
        except NetStringClosed as e:
            assert str(e) == 'connection closed'
