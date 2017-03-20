import time
import numpy as np

import logging
log = logging.getLogger(__name__)

class SampleAgent(object):

    def __init__(self,parameter_server):
        self.ps = parameter_server

    def init(self,ignore=None):
        log.info("initialize agent: load NN, do other initialization steps as needed")
        
        if self._isSleep(): 
            time.sleep(0.5)
        
        return self._isOK()

    def update(self,data):
        log.info("processing update: "+str(data))
        
        if self._isSleep(): 
            time.sleep(0.02)
        
        action = self.ps.run(ops=['act'], 
            feed_dict={'state': np.array(data['state'])})
        
        log.info("global step:"+str(self.ps.run(ops=['step'],feed_dict={})[0]))

        return action[0]

    def reset(self,ignore=None):
        log.info("reseting agent")
        
        if self._isSleep(): 
            time.sleep(0.01)
        
        return self._isOK()     

    def _isOK(self):
        return True 

    def _isSleep(self):
        return False
