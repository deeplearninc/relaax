import io
import sys
import json
import base64
import numpy as np

class RLXMessage():
    class _NDArrayEncoder(json.JSONEncoder):
        def default(self, obj):
            # Encoder from numpy.nparray to base64
            if isinstance(obj, np.ndarray):
                output = io.BytesIO()
                np.savez_compressed(output, obj=obj)
                return {'b64npz': base64.b64encode(output.getvalue())}
            return json.JSONEncoder.default(self, obj)

    @staticmethod
    def _ndarray_decoder(dct):
        # Decoder from base64 to numpy.ndarray
        if isinstance(dct, dict) and 'b64npz' in dct:
            output = io.BytesIO(base64.b64decode(dct['b64npz']))
            output.seek(0)
            return np.load(output)['obj']
        return dct

    @classmethod
    def _encode_array(self,arr):
        # Server expects numpy.nparray so 
        # always encodeing as numpy.nparray
        if isinstance(arr, np.ndarray):
            data = arr
        else:
            data = np.asarray(arr,dtype=np.float32)
        return json.dumps(data,cls=self._NDArrayEncoder)

    @classmethod
    def client_to_wire(self,verb,state=None,reward=None):
        message = [verb]
        if reward:
            message.append(reward)
        if state:
            message.append(self._encode_array(state))
        data = json.dumps(message)
        return '%d:%s,' % (len(data), data)

    @classmethod
    def client_from_wire(self,wire_data,to_parray=False):
        message = {}
        data = json.loads(wire_data,object_hook=self._ndarray_decoder)
        message['verb'] = data[0]
        if message['verb'] == 'act':
            message['action'] = data[1].tolist() if to_parray else data[1]
        return message




