import re


class NetStringException(Exception):
    pass


class NetStringClosed(Exception):
    pass


class NetString():
    MAX_STRING_LEN = 10**9
    MAX_LEN_DIGITS = 10

    def __init__(self, skt):
        self.skt = skt

    def readString(self):
        slen = self._receive_length()
        if slen > self.MAX_STRING_LEN:
            raise NetStringException("net string too long")
        s = self._receiveb(slen)
        if self._receiveb(1) != ',':
            raise NetStringException("wrong net string format")
        return s

    def writeString(self, data):
        try:
            if len(data) > self.MAX_STRING_LEN:
                raise NetStringException("can't send, net string too long")
            self.skt.sendall('%d:%s,' % (len(data), data))
        except NetStringException as e:
            raise e
        except:
            raise NetStringClosed("net string closed")

    def _receiveb(self, length):
        packets = []
        rest = length
        while rest > 0:
            packet = self.skt.recv(rest)
            if not packet:
                raise NetStringClosed("can't receive, net string closed")
            packets.append(packet)
            rest -= len(packet)
        data = ''.join(packets)
        assert len(data) == length
        return data

    def _receive_length(self):
        digits = []

        while True:
            char = self.skt.recv(1)
            if char == ':':
                break
            if not char:
                raise NetStringClosed("net string closed")
            if (re.match('^\d$', char) is None) or \
               (len(digits) > 0 and digits[0] == 0) or \
               (len(digits) > self.MAX_LEN_DIGITS):
                raise NetStringException(
                    "can't receive, wrong net string format")
            digits.append(char)

        return int(''.join(digits))
