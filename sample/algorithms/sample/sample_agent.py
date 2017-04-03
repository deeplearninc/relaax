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
        log.info("processing state: " + str(state))

        if self._is_sleep():
            time.sleep(0.02)

        action = self.ps.run(
            [self.ps.model.act],
            feed_dict={self.ps.model.state: np.array(state)}
        )

        log.info("global step:" +
                 str(self.ps.run([self.ps.model.step])[0]))

        return action[0]

    def reset(self, ignore=None):
        log.info("reseting agent")

        if self._is_sleep():
            time.sleep(0.01)

        return self._is_ok()

    def _is_ok(self):
        return True

    def _is_sleep(self):
        return False
