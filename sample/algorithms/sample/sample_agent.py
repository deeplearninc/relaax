from builtins import str
from builtins import object
import time
import logging
import numpy as np

log = logging.getLogger(__name__)


class SampleAgent(object):

    def __init__(self, parameter_server):
        self.ps = parameter_server

    def init(self, ignore=None):
        log.info("initialize agent: load NN, do other initialization steps")

        if self._is_sleep():
            time.sleep(0.5)

        return self._is_ok()

    def update(self, reward, state, terminal):
        # log.info("processing state: " + str(state))

        if self._is_sleep():
            time.sleep(0.02)

        if terminal:
            action = None
        else:
            action = self.ps.op_act(state=np.array(state))

        log.info("global step:" + str(self.ps.op_step()))

        return action

    def reset(self, ignore=None):
        log.info("reseting agent")

        if self._is_sleep():
            time.sleep(0.01)

        return self._is_ok()

    def _is_ok(self):
        return True

    def _is_sleep(self):
        return False
