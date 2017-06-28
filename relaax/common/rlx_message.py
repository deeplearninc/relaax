from builtins import object
import io
import json
import base64
import numpy as np
from PIL import Image
from struct import *

class RLXMessageImage(object):
    def __init__(self, image):
        self.image = image #np.asarray(image)

class RLXMessage(object):
    #types: https://docs.python.org/3.5/library/struct.html#struct.pack_into
    TYPE_NONE = 0
    TYPE_NULL = 1
    TYPE_INT4 = 2
    TYPE_STRING_UTF8 = 3
    TYPE_DOUBLE = 4
    TYPE_BOOLEAN = 5
    TYPE_IMAGE = 6
    TYPE_NDARRAY = 7
    TYPE_LIST = 8
    TYPE_UINT4 = 9

    # class _NDArrayEncoder(json.JSONEncoder):
    #     def default(self, obj):
    #         # Encoder from numpy.nparray to base64
    #         if isinstance(obj, np.ndarray):
    #             output = io.BytesIO()
    #             np.savez_compressed(output, obj=obj)
    #             return {'b64npz': base64.b64encode(output.getvalue()).decode()}
    #         return json.JSONEncoder.default(self, obj)
    #
    # @staticmethod
    # def _ndarray_decoder(obj):
    #     # Decoder from base64 to numpy.ndarray
    #     if isinstance(obj, dict) and 'b64npz' in obj:
    #         output = io.BytesIO(base64.b64decode(obj['b64npz']))
    #         output.seek(0)
    #         return np.load(output)['obj']
    #     elif isinstance(obj, RLXMessageImage):
    #         return np.array(obj.image.convert("RGB"))
    #
    #     return obj

    @classmethod
    def _pack_string(cls, value):
        buf = bytearray()
        bval = bytearray(value.encode('UTF-8'))
        buf += pack("I", len(bval))
        buf += bval

        return buf

    @classmethod
    def _pack_value(cls, value):
        buf = bytearray()
        if isinstance(value, bool):
            buf += pack("B", cls.TYPE_BOOLEAN)
            buf += pack("B", 1 if value else 0)
        elif isinstance(value, int):
            buf += pack("B", cls.TYPE_INT4)
            buf += pack("i", value)
        elif isinstance(value, float):
            buf += pack("B", cls.TYPE_DOUBLE)
            buf += pack("d", value)
        elif isinstance(value, str):
            buf += pack("B", cls.TYPE_STRING_UTF8)
            buf += cls._pack_string(value)
        elif value is None:
            buf += pack("B", cls.TYPE_NULL)
        elif isinstance(value, RLXMessageImage):
            buf += pack("B", cls.TYPE_IMAGE)
            buf += cls._pack_string(value.image.mode)
            buf += pack("I", value.image.size[0])
            buf += pack("I", value.image.size[1])

            bval = value.image.tobytes()
            buf += pack("I", len(bval))
            buf += bval
        elif isinstance(value, np.ndarray):
            buf += pack("B", cls.TYPE_NDARRAY)
            buf += cls._pack_string(str(value.dtype))
            buf += pack("I", len(value.shape))
            for ns in value.shape:
                buf += pack("I", ns)

            bval = value.tobytes()
            buf += pack("I", len(bval))
            buf += bval
        elif isinstance(value, list):
            buf += pack("B", cls.TYPE_LIST)
            buf += pack("I", len(value))
            for item in value:
                buf += cls._pack_value(item)
        else:
            print("Pack Unknown type:"+str(type(value)))
            raise Exception("Pack Unknown type:"+str(type(value)))
        return buf

    @classmethod
    def _unpack_string(cls, buf, offset):
        reslen = unpack_from("I", buf, offset)[0]
        offset += 4
        res = str(buf[offset:offset+reslen].decode('UTF-8'))
        offset += reslen

        return res, offset

    @classmethod
    def _unpack_value(cls, buf, offset):

        valtype = unpack_from("B", buf, offset)[0]
        offset += 1
        res = None
        #print(valtype)
        if valtype == cls.TYPE_INT4:
            res = unpack_from("i", buf, offset)[0]
            offset += 4
        elif valtype == cls.TYPE_DOUBLE:
            res = unpack_from("d", buf, offset)[0]
            offset += 8
        elif valtype == cls.TYPE_STRING_UTF8:
            (res, offset) = cls._unpack_string(buf, offset)
        elif valtype == cls.TYPE_BOOLEAN:
            res = unpack_from("B", buf, offset)[0]
            res = True if res == 1 else False
            offset += 1
        elif valtype == cls.TYPE_NULL:
            res = None
        elif valtype == cls.TYPE_IMAGE:
            (mode, offset) = cls._unpack_string(buf, offset)
            x = unpack_from("I", buf, offset)[0]
            offset += 4
            y = unpack_from("I", buf, offset)[0]
            offset += 4

            reslen = unpack_from("I", buf, offset)[0]
            offset += 4
            img = Image.frombytes(mode, (x, y), bytes(buf[offset:offset+reslen])).convert("RGB")
            res = np.asarray(img)
            offset += reslen
        elif valtype == cls.TYPE_NDARRAY:
            (dtype, offset) = cls._unpack_string(buf, offset)
            shape_len = unpack_from("I", buf, offset)[0]
            offset += 4
            shape = []
            for i in range(0, shape_len):
                item = unpack_from("I", buf, offset)[0]
                offset += 4
                shape.append(item)

            reslen = unpack_from("I", buf, offset)[0]
            offset += 4
            res = np.frombuffer(buf[offset:offset+reslen], dtype=np.dtype(dtype) ) #, count=reslen, offset=offset)
            res = res.reshape(shape)
            offset += reslen
        elif valtype == cls.TYPE_LIST:
            reslen = unpack_from("I", buf, offset)[0]
            offset += 4
            res = []
            for i in range(0, reslen):
                (item, offset) = cls._unpack_value(buf, offset)
                res.append(item)
        else:
            print("Unknown type:%d" % valtype)
            raise Exception("Unknown type:%d" % valtype)

        return res, offset

    @classmethod
    def to_wire(cls, data):
        #print("To_wire:"+str(data))
        buf = bytearray()#json.dumps(data, cls=cls._NDArrayEncoder)#bytearray()

        for key in data:
            buf += cls._pack_string(key)
            #print((key, type(data[key])))
            buf += cls._pack_value(data[key])

        return buf

    @classmethod
    def from_wire(cls, data):
        res = {}#json.loads(data, object_hook=cls._ndarray_decoder) #{}
        offset = 0

        while offset < len(data):
            (key, offset) = cls._unpack_string(data, offset)
            (res[key], offset) = cls._unpack_value(data, offset)

        #print("From wire:"+str(res))
        # for key in res:
        #     print((key,type(res[key])))

        return res

# may be return an object instead of dict
# def from_wire():
#   class Object(object):
#     pass
#   robj = Object()
#   iterate through json and do
#   setattr(robj, key, value)
#   return robj
