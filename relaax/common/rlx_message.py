from builtins import object

import numpy as np
from PIL import Image
from struct import pack, unpack_from

V = False


class RLXMessageImage(object):
    def __init__(self, image):
        self.image = image


class RLXMessage(object):
    # types: https://docs.python.org/3.5/library/struct.html#struct.pack_into
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
    TYPE_INT64 = 10
    TYPE_DICT = 11

    pack_info = {
        type(None).__name__: {'id': TYPE_NULL, 'pack_func': lambda cls, *args: cls._pack_null(*args),
                              'unpack_func': lambda cls, *args: cls._unpack_null(*args)},
        int.__name__: {'id': TYPE_INT4, 'pack_func': lambda cls, *args: cls._pack_int(*args),
                       'unpack_func': lambda cls, *args: cls._unpack_int(*args)},
        'int32': {'id': TYPE_INT4, 'pack_func': lambda cls, *args: cls._pack_int(*args),
                  'unpack_func': lambda cls, *args: cls._unpack_int(*args)},
        'int64': {'id': TYPE_INT64, 'pack_func': lambda cls, *args: cls._pack_int64(*args),
                  'unpack_func': lambda cls, *args: cls._unpack_int64(*args)},
        str.__name__: {'id': TYPE_STRING_UTF8, 'pack_func': lambda cls, *args: cls._pack_string(*args),
                       'unpack_func': lambda cls, *args: cls._unpack_string(*args)},
        float.__name__: {'id': TYPE_DOUBLE, 'pack_func': lambda cls, *args: cls._pack_double(*args),
                         'unpack_func': lambda cls, *args: cls._unpack_double(*args)},
        'float64': {'id': TYPE_DOUBLE, 'pack_func': lambda cls, *args: cls._pack_double(*args),
                    'unpack_func': lambda cls, *args: cls._unpack_double(*args)},
        bool.__name__: {'id': TYPE_BOOLEAN, 'pack_func': lambda cls, *args: cls._pack_bool(*args),
                        'unpack_func': lambda cls, *args: cls._unpack_bool(*args)},
        RLXMessageImage.__name__: {'id': TYPE_IMAGE, 'pack_func': lambda cls, *args: cls._pack_image(*args),
                                   'unpack_func': lambda cls, *args: cls._unpack_image(*args)},
        np.ndarray.__name__: {'id': TYPE_NDARRAY, 'pack_func': lambda cls, *args: cls._pack_ndarray(*args),
                              'unpack_func': lambda cls, *args: cls._unpack_ndarray(*args)},
        list.__name__: {'id': TYPE_LIST, 'pack_func': lambda cls, *args: cls._pack_list(*args),
                        'unpack_func': lambda cls, *args: cls._unpack_list(*args)},
        dict.__name__: {'id': TYPE_DICT, 'pack_func': lambda cls, *args: cls._pack_dict(*args),
                        'unpack_func': lambda cls, *args: cls._unpack_dict(*args)}
    }

    @classmethod
    def _pack_type(cls, type_id, buf, pack_type=True):
        if pack_type:
            buf += pack("B", type_id)

    @classmethod
    def _unpack_type(cls, buf, offset):
        return unpack_from("B", buf, offset)[0], offset+1

    @classmethod
    def _pack_null(cls, value, buf, pack_type=True):
        cls._pack_type(cls.TYPE_NULL, buf, pack_type)

    @classmethod
    def _unpack_null(cls, buf, offset):
        return None, offset

    @classmethod
    def _pack_string(cls, value, buf, pack_type=True):
        cls._pack_type(cls.TYPE_STRING_UTF8, buf, pack_type)

        bval = bytearray(str(value).encode('UTF-8'))
        buf += pack("I", len(bval))
        buf += bval

    @classmethod
    def _unpack_string(cls, buf, offset):
        reslen = unpack_from("I", buf, offset)[0]
        offset += 4
        res = str(buf[offset:offset+reslen].decode('UTF-8'))
        offset += reslen

        return res, offset

    @classmethod
    def _pack_int(cls, value, buf, pack_type=True):
        cls._pack_type(cls.TYPE_INT4, buf, pack_type)
        buf += pack("i", value)

    @classmethod
    def _unpack_int(cls, buf, offset):
        res = unpack_from("i", buf, offset)[0]
        offset += 4
        return res, offset

    @classmethod
    def _pack_int64(cls, value, buf, pack_type=True):
        cls._pack_type(cls.TYPE_INT64, buf, pack_type)
        buf += pack("q", value)

    @classmethod
    def _unpack_int64(cls, buf, offset):
        res = unpack_from("q", buf, offset)[0]
        offset += 8
        return res, offset

    @classmethod
    def _pack_double(cls, value, buf, pack_type=True):
        cls._pack_type(cls.TYPE_DOUBLE, buf, pack_type)
        buf += pack("d", value)

    @classmethod
    def _unpack_double(cls, buf, offset):
        res = unpack_from("d", buf, offset)[0]
        offset += 8
        return res, offset

    @classmethod
    def _pack_bool(cls, value, buf, pack_type=True):
        cls._pack_type(cls.TYPE_BOOLEAN, buf, pack_type)
        buf += pack("B", 1 if value else 0)

    @classmethod
    def _unpack_bool(cls, buf, offset):
        res = unpack_from("B", buf, offset)[0]
        res = res == 1
        offset += 1
        return res, offset

    @classmethod
    def _pack_image(cls, value, buf, pack_type=True):
        cls._pack_type(cls.TYPE_IMAGE, buf, pack_type)
        cls._pack_string(value.image.mode, buf, False)
        buf += pack("I", value.image.size[0])
        buf += pack("I", value.image.size[1])

        bval = value.image.tobytes()
        buf += pack("I", len(bval))
        buf += bval

    @classmethod
    def _unpack_image(cls, buf, offset):
        mode, offset = cls._unpack_string(buf, offset)
        x = unpack_from("I", buf, offset)[0]
        offset += 4
        y = unpack_from("I", buf, offset)[0]
        offset += 4

        reslen = unpack_from("I", buf, offset)[0]
        offset += 4
        img = Image.frombytes(mode, (x, y), bytes(buf[offset:offset+reslen]))  # .convert("RGB")
        res = np.asarray(img)
        if img.mode in ["L", "RGB", "RGBA", "CMYK", "YCbCr", "LAB", "HSV"]:
            res = res.astype(np.float32) * (1.0 / 255.0)

        # print(res.shape)
        # res = np.reshape(res, (x, y, 1))
        offset += reslen
        return res, offset

    @classmethod
    def _pack_ndarray(cls, value, buf, pack_type=True):
        cls._pack_type(cls.TYPE_NDARRAY, buf, pack_type)
        cls._pack_string(str(value.dtype), buf, False)
        buf += pack("I", len(value.shape))
        for ns in value.shape:
            buf += pack("I", ns)

        bval = value.tobytes()
        buf += pack("I", len(bval))
        buf += bval

    @classmethod
    def _unpack_ndarray(cls, buf, offset):
        dtype, offset = cls._unpack_string(buf, offset)
        shape_len = unpack_from("I", buf, offset)[0]
        offset += 4
        shape = []
        for i in range(0, shape_len):
            item = unpack_from("I", buf, offset)[0]
            offset += 4
            shape.append(item)

        reslen = unpack_from("I", buf, offset)[0]
        offset += 4
        res = np.frombuffer(buf[offset:offset+reslen], dtype=np.dtype(dtype))
        res = res.reshape(shape)
        offset += reslen
        return res, offset

    @classmethod
    def _pack_list(cls, value, buf, pack_type=True):
        cls._pack_type(cls.TYPE_LIST, buf, pack_type)
        buf += pack("I", len(value))
        for item in value:
            cls._pack_value(item, buf)

    @classmethod
    def _unpack_list(cls, buf, offset):
        reslen = unpack_from("I", buf, offset)[0]
        offset += 4
        res = []
        for i in range(0, reslen):
            (item, offset) = cls._unpack_value(buf, offset)
            res.append(item)
        return res, offset

    @classmethod
    def _pack_dict(cls, value, buf, pack_type=True):
        cls._pack_type(cls.TYPE_DICT, buf, pack_type)
        buf += pack("I", len(value))

        for key in value:
            cls._pack_string(key, buf, False)
            cls._pack_value(value[key], buf)

    @classmethod
    def _unpack_dict(cls, buf, offset):
        reslen = unpack_from("I", buf, offset)[0]
        offset += 4
        res = {}
        for i in range(0, reslen):
            (field_name, offset) = cls._unpack_string(buf, offset)
            (item, offset) = cls._unpack_value(buf, offset)
            res[field_name] = item
        return res, offset

    @classmethod
    def _pack_value(cls, value, buf):
        item = cls.pack_info.get(type(value).__name__, None)
        if item is None:
            item = cls.pack_info.get(str.__name__)

        item.get('pack_func')(cls, value, buf)

    @classmethod
    def _unpack_value(cls, buf, offset):
        valtype, offset = cls._unpack_type(buf, offset)
        for key, item in cls.pack_info.items():
            if item['id'] == valtype:
                return item['unpack_func'](cls, buf, offset)

        raise Exception("Unknown type:%d" % valtype)

    @classmethod
    def to_wire(cls, data):
        buf = bytearray()
        if V:
            buf += pack("I", 1)

        for key in data:
            # print((key, type(data[key]).__name__))
            cls._pack_string(key, buf, False)
            cls._pack_value(data[key], buf)

        return buf

    @classmethod
    def from_wire(cls, data):
        res = {}
        offset = 0
        if V:
            # read version
            unpack_from("I", data, offset)[0]
            offset += 4

        while offset < len(data):
            (field_name, offset) = cls._unpack_string(data, offset)
            (res[field_name], offset) = cls._unpack_value(data, offset)

        return res
