import threading

import numpy as np
import io
import base64


class BaseModel(threading.Thread):
    def __init__(self, sio, session, namespace):
        self.sio = sio
        self.session = session
        self.namespace = namespace
        self._stop = threading.Event()
        threading.Thread.__init__(self)

    def run(self):
        print("Sending 'model is allocated' message to room: " + self.session)
        self.sio.emit('model is allocated', {}, room=self.session, namespace=self.namespace)
        self.train()
        print("Model for " + self.session + " stopped")
        self.sio.emit('model is stopped', {'data': self.session}, room=self.session, namespace=self.namespace)

    def stop(self):
        self._stop.set()

    def isStopped(self):
        return self._stop.isSet()

    @staticmethod
    def ndarray_decoder(dct):
        """Decoder from base64 to np.ndarray for big arrays(states)"""
        if isinstance(dct, dict) and 'b64npz' in dct:
            output = io.BytesIO(base64.b64decode(dct['b64npz']))
            output.seek(0)
            return np.load(output)['obj']
        return dct

    @staticmethod
    def to_camelcase(value):
        return ''.join(x.capitalize() or '_' for x in value.split('_'))

    def getAction(self, message):
        pass

    def addEpisode(self, episode):
        pass

    def train(self):
        pass
