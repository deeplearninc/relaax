import numpy
import types

import bridge_pb2


class BridgeProtocol(object):

    @staticmethod
    def build_messages(value):
        return BridgeProtocol.build_messages_recursive(value, dict_key=None)

    @staticmethod
    def parse_messages(messages):
        value, _ = BridgeProtocol.parse_messages_recursive(next(messages), messages)
        the_end = object()
        assert next(messages, the_end) == the_end
        return value

    @staticmethod
    def build_messages_recursive(value, dict_key):
        return BridgeProtocol.BUILD_ITEM_MESSAGES_BY_TYPE[type(value)](value, dict_key)

    def build_messages_for_list(value, dict_key):
        yield bridge_pb2.Item(item_type=bridge_pb2.Item.LIST_OPEN)
        for item in value:
            for message in BridgeProtocol.build_messages_recursive(item, dict_key=dict_key):
                yield message
        yield bridge_pb2.Item(item_type=bridge_pb2.Item.LIST_CLOSE, dict_key=dict_key)

    def build_messages_for_dict(value, dict_key):
        yield bridge_pb2.Item(item_type=bridge_pb2.Item.DICT_OPEN)
        for key, item in value.iteritems():
            for message in BridgeProtocol.build_messages_recursive(item, dict_key=key):
                yield message
        yield bridge_pb2.Item(item_type=bridge_pb2.Item.DICT_CLOSE, dict_key=dict_key)

    def build_messages_for_none(value, dict_key):
        yield bridge_pb2.Item(item_type=bridge_pb2.Item.NONE, dict_key=dict_key)

    def build_messages_for_bool(value, dict_key):
        yield bridge_pb2.Item(item_type=bridge_pb2.Item.BOOL, dict_key=dict_key, bool_value=value)

    def build_messages_for_int(value, dict_key):
        yield bridge_pb2.Item(item_type=bridge_pb2.Item.INT, dict_key=dict_key, int_value=value)

    def build_messages_for_float(value, dict_key):
        yield bridge_pb2.Item(item_type=bridge_pb2.Item.FLOAT, dict_key=dict_key, float_value=value)

    def build_messages_for_str(value, dict_key):
        yield bridge_pb2.Item(item_type=bridge_pb2.Item.STR, dict_key=dict_key, str_value=value)

    def build_messages_for_ndarray(array, dict_key):
        # TODO: select more appropriate block size
        block_size = 1024 * 1024

        dtype = str(array.dtype)
        shape = array.shape
        data = array.data
        size = len(data)

        # optimization to avoid extra data copying if array data fits to one block
        # TODO: compare actual performance
        if size <= block_size:
            bytes_ = array.tobytes()
            assert size == len(bytes_)
            yield bridge_pb2.Item(
                item_type=bridge_pb2.Item.NUMPY_ARRAY,
                dict_key=dict_key,
                numpy_array_value=bridge_pb2.Item.NumpyArray(
                    last=True,
                    dtype=dtype,
                    shape=shape,
                    data=bytes_
                )
            )
        else:
            i = 0
            while i < size:
                ii = i + block_size
                if ii >= size:
                    yield bridge_pb2.Item(
                        item_type=bridge_pb2.Item.NUMPY_ARRAY,
                        dict_key=dict_key,
                        numpy_array_value=bridge_pb2.Item.NumpyArray(
                            last=True,
                            dtype=dtype,
                            shape=shape,
                            data=data[i:ii]
                        )
                    )
                else:
                    yield bridge_pb2.Item(
                        item_type=bridge_pb2.Item.NUMPY_ARRAY,
                        numpy_array_value=bridge_pb2.Item.NumpyArray(
                            last=False,
                            data=data[i:ii]
                        )
                    )
                i = ii

    @staticmethod
    def parse_messages_recursive(message, messages):
        return BridgeProtocol.PARSE_ITEM_MESSAGES_BY_TYPE[message.item_type](message, messages)

    def parse_messages_for_list(message, messages):
        value = []
        while True:
            message = next(messages)
            if message.item_type == bridge_pb2.Item.LIST_CLOSE:
                return value, message
            value.append(BridgeProtocol.parse_messages_recursive(message, messages)[0])

    def parse_messages_for_dict(message, messages):
        value = {}
        while True:
            message = next(messages)
            if message.item_type == bridge_pb2.Item.DICT_CLOSE:
                return value, message
            item, last_message = BridgeProtocol.parse_messages_recursive(message, messages)
            value[last_message.dict_key] = item

    def parse_messages_for_none(message, messages):
        return None, message

    def parse_messages_for_bool(message, messages):
        return message.bool_value, message

    def parse_messages_for_int(message, messages):
        return int(message.int_value), message

    def parse_messages_for_float(message, messages):
        return message.float_value, message

    def parse_messages_for_str(message, messages):
        return str(message.str_value), message

    def parse_messages_for_ndarray(message, messages):
        data = []
        while True:
            if message.numpy_array_value.last:
                # optimization to avoid extra data copying if array data fits to one block
                # TODO: compare actual performance
                if len(data) == 0:
                    buffer_ = message.numpy_array_value.data
                else:
                    data.append(message.numpy_array_value.data)
                    buffer_ = ''.join(data)

                value = numpy.ndarray(
                    shape=message.numpy_array_value.shape,
                    dtype=numpy.dtype(message.numpy_array_value.dtype),
                    buffer=buffer_
                )
                return value, message
            else:
                data.append(message.numpy_array_value.data)
            message = next(messages)
            assert message.item_type == bridge_pb2.Item.NUMPY_ARRAY

    BUILD_ITEM_MESSAGES_BY_TYPE = {
        list          : build_messages_for_list   ,
        dict          : build_messages_for_dict   ,
        types.NoneType: build_messages_for_none   ,
        bool          : build_messages_for_bool   ,
        int           : build_messages_for_int    ,
        float         : build_messages_for_float  ,
        str           : build_messages_for_str    ,
        numpy.ndarray : build_messages_for_ndarray
    }

    PARSE_ITEM_MESSAGES_BY_TYPE = {
        bridge_pb2.Item.LIST_OPEN  : parse_messages_for_list   ,
        bridge_pb2.Item.DICT_OPEN  : parse_messages_for_dict   ,
        bridge_pb2.Item.NONE       : parse_messages_for_none   ,
        bridge_pb2.Item.BOOL       : parse_messages_for_bool   ,
        bridge_pb2.Item.INT        : parse_messages_for_int    ,
        bridge_pb2.Item.FLOAT      : parse_messages_for_float  ,
        bridge_pb2.Item.STR        : parse_messages_for_str    ,
        bridge_pb2.Item.NUMPY_ARRAY: parse_messages_for_ndarray
    }
