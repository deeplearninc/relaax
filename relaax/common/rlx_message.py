from builtins import object
import io
import json
import base64
import numpy as np


class RLXMessage(object):
    class _NDArrayEncoder(json.JSONEncoder):
        def default(self, obj):
            # Encoder from numpy.nparray to base64
            if isinstance(obj, np.ndarray):
                output = io.BytesIO()
                np.savez_compressed(output, obj=obj)
                return {'b64npz': base64.b64encode(output.getvalue()).decode()}
            return json.JSONEncoder.default(self, obj)

    @staticmethod
    def _ndarray_decoder(obj):
        # Decoder from base64 to numpy.ndarray
        if isinstance(obj, dict) and 'b64npz' in obj:
            output = io.BytesIO(base64.b64decode(obj['b64npz']))
            output.seek(0)
            return np.load(output)['obj']
        return obj

    @classmethod
    def to_wire(cls, data):
        return json.dumps(data, cls=cls._NDArrayEncoder).encode()

    @classmethod
    def from_wire(cls, data):
        return json.loads(data.decode(), object_hook=cls._ndarray_decoder)

# may be return an object instead of dict
# def from_wire():
#   class Object(object):
#     pass
#   robj = Object()
#   iterate through json and do
#   setattr(robj, key, value)
#   return robj
