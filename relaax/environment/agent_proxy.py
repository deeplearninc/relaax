from builtins import str
from builtins import object
import time
import socket
import random
import logging

from relaax.common.rlx_netstring import NetString
from relaax.common.rlx_message import RLXMessage as rlxm

log = logging.getLogger(__name__)


class AgentProxyException(Exception):
    pass


class AgentProxy(object):

    def __init__(self, rlx_server_url):
        self.skt = None
        self.transport = None
        self.address = rlx_server_url
        self.metrics = AgentProxyMetrics(self._update_metrics)

    def init(self, exploit=False):
        return self._exchange({'command': 'init', 'exploit': exploit})

    def update(self, reward=None, state=None, terminal=False):
        return self._exchange({
            'command': 'update',
            'reward': reward,
            'state': state,
            'terminal': terminal})

    def reset(self):
        return self._exchange({'command': 'reset'})

    def _update_metrics(self, name, y, x=None):
        return self._exchange({
            'command': 'update_metrics',
            'name': name,
            'y': y,
            'x': x
        })

    def _exchange(self, data):
        if self.skt:
            try:
                self.transport.write_string(rlxm.to_wire(data))
                ret = rlxm.from_wire(self.transport.read_string())
            except Exception as e:
                raise AgentProxyException(str(e))
            if not ('response' in ret):
                raise AgentProxyException("wring message format")
            elif ret['response'] == 'error':
                raise AgentProxyException(
                    ret['message'] if 'message' in ret else "unknown error")
        else:
            raise AgentProxyException("no connection is available.")
        return ret['data'] if 'data' in ret else ret['response']

    def connect(self, retry=6):

        self.disconnect()

        def parse_address(address):
            try:
                host, port = address.split(':')
                return host, int(port)
            except Exception:
                raise AgentProxyException("can't parse server address.")

        if not isinstance(self.address, tuple):
            self.address = parse_address(self.address)

        count = 0
        random.seed()
        while True:
            try:
                log.info("trying to connect, attempt: %d (%d)" % (count + 1, retry))
                s = socket.socket()
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.connect(self.address)
                self.transport = NetString(s)
                assert s is not None
                self.skt = s
                break
            except socket.error as e:
                s.close()
                count += 1
                if count < retry:
                    time.sleep(random.uniform(0.5, 6))
                    continue
                else:
                    raise AgentProxyException(str(e))
            except Exception as e:
                raise AgentProxyException(str(e))

    def disconnect(self):
        if self.skt:
            self.skt.close()
            self.skt = None


class AgentProxyMetrics(object):
    def __init__(self, send_metrics):
        self._update_metrics = send_metrics

    def scalar(self, name, y, x=None):
        self._update_metrics(name, y, x)
