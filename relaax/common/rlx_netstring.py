from builtins import str
from builtins import object
import re
from struct import *

class NetStringException(Exception):
    pass


class NetStringClosed(Exception):
    pass


class NetString(object):
    MAX_STRING_LEN = 10**9
    MAX_LEN_DIGITS = len(str(MAX_STRING_LEN - 1))

    def __init__(self, skt):
        self.skt = skt

    def read_string(self):
        slen = self._receive_length()
        if slen > self.MAX_STRING_LEN:
            raise NetStringException("net string too long")
        s = self._receiveb(slen)
        if self._receiveb(1) != ',':
            raise NetStringException("wrong net string format")
        return s

    def write_string(self, data):
        try:
            if len(data) > self.MAX_STRING_LEN:
                raise NetStringException("can't send, net string too long")

            self.skt.sendall(('%d:' % (len(data))).encode())
            self.skt.sendall(data)
            self.skt.sendall(','.encode())

        except NetStringException as e:
            print(str(e))
            raise e
        except Exception as e1:
            print(str(e1))
            raise NetStringClosed("connection closed")

    def _receiveb(self, length):
        packets = []
        rest = length
        while rest > 0:
            packet = self.skt.recv(rest)
            if not packet:
                raise NetStringClosed("can't receive, net string closed")
            packets.append(packet)
            rest -= len(packet)
        data = b''.join(packets)
        assert len(data) == length
        return data#.decode()

    def _receive_length(self):
        digits = []

        while True:
            b = self.skt.recv(1)
            char = b.decode()
            if char == ':':
                break
            if not char:
                raise NetStringClosed("connection closed")
            if (re.match('^\d$', char) is None) or \
               (len(digits) > 0 and digits[0] == 0) or \
               (len(digits) >= self.MAX_LEN_DIGITS):
                raise NetStringException("can't receive, wrong net string format")
            digits.append(char)

        return int(''.join(digits))
