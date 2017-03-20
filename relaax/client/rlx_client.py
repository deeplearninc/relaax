import socket, time, random
from relaax.common.rlx_netstring import NetString
from relaax.common.rlx_message import RLXMessage as rlxm

import logging
log = logging.getLogger(__name__)

class RlxClientException(Exception):
    pass

class RlxClient(object):

    def __init__(self,rlx_server_url):
        self.skt = None
        self.transport = None
        self.address = rlx_server_url

    def init(self):
        return self._exchange({'command':'init'})

    def update(self,reward=None,state=None,terminal=False):
        return self._exchange({'command':'update',
            'reward':reward,'state':state,'terminal':terminal})

    def reset(self):
        return self._exchange({'command':'reset'})

    def _exchange(self,data):
        if self.skt:
            try:
                self.transport.writeString(rlxm.to_wire(data))
                ret = rlxm.from_wire(self.transport.readString())
            except Exception as e:
                raise RlxClientException(str(e))
            if ret['response'] == 'error':
                raise RlxClientException(ret['message'])
        else:
            raise RlxClientException("no connection is available.")
        return ret

    def connect(self,retry=3):

        self.disconnect()

        def parse_address(address):
            try:
                host, port = address.split(':')
                return host, int(port)
            except Exception as e:
                raise RlxClientException("can't parse server address.")

        if not isinstance(self.address,tuple):
            self.address = parse_address(self.address)

        count = 0
        random.seed()
        while True:
            try:
                log.info("trying to connect, attempt: %d (%d)"%(count+1,retry))
                s = socket.socket()
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.connect(self.address)
                self.transport = NetString(s)
                assert s is not None
                self.skt = s
                break
            except socket.error as e:
                if s is not None:
                    s.close()
                count += 1
                if count < retry:
                    time.sleep(random.uniform(0.5,6))
                    continue
                else:
                    raise RlxClientException(str(e))
            except Exception as e:
                raise RlxClientException(str(e))

    def disconnect(self):
        if self.skt:
            self.skt.close()
            self.skt = None
