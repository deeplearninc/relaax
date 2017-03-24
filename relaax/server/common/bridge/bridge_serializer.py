import numpy
import types

import bridge_pb2


class BridgeSerializer(object):

    @staticmethod
    def serialize(value):
        return BridgeSerializer.serialize_recursive(value, dict_key=None)

    @staticmethod
    def deserialize(messages):
        value, _ = BridgeSerializer.deserialize_recursive(next(messages), messages)
        the_end = object()
        assert next(messages, the_end) == the_end
        return value

    @staticmethod
    def serialize_recursive(value, dict_key):
        return BridgeSerializer.SERIALIZERS[type(value)](value, dict_key)

    @staticmethod
    def deserialize_recursive(message, messages):
        return BridgeSerializer.DESERIALIZERS[message.item_type](message, messages)

    def serialize_list(value, dict_key):
        yield bridge_pb2.Item(item_type=bridge_pb2.Item.LIST_OPEN)
        for item in value:
            for message in BridgeSerializer.serialize_recursive(item, dict_key=dict_key):
                yield message
        yield bridge_pb2.Item(item_type=bridge_pb2.Item.LIST_CLOSE, dict_key=dict_key)

    def deserialize_list(message, messages):
        value = []
        while True:
            message = next(messages)
            if message.item_type == bridge_pb2.Item.LIST_CLOSE:
                return value, message
            value.append(BridgeSerializer.deserialize_recursive(message, messages)[0])

    def serialize_dict(value, dict_key):
        yield bridge_pb2.Item(item_type=bridge_pb2.Item.DICT_OPEN)
        for key, item in value.iteritems():
            for message in BridgeSerializer.serialize_recursive(item, dict_key=key):
                yield message
        yield bridge_pb2.Item(item_type=bridge_pb2.Item.DICT_CLOSE, dict_key=dict_key)

    def deserialize_dict(message, messages):
        value = {}
        while True:
            message = next(messages)
            if message.item_type == bridge_pb2.Item.DICT_CLOSE:
                return value, message
            item, last_message = BridgeSerializer.deserialize_recursive(message, messages)
            value[last_message.dict_key] = item

    def serialize_none(value, dict_key):
        yield bridge_pb2.Item(item_type=bridge_pb2.Item.NONE, dict_key=dict_key)

    def deserialize_none(message, messages):
        return None, message

    def serialize_bool(value, dict_key):
        yield bridge_pb2.Item(item_type=bridge_pb2.Item.BOOL, dict_key=dict_key, bool_value=value)

    def deserialize_bool(message, messages):
        return message.bool_value, message

    def serialize_int(value, dict_key):
        yield bridge_pb2.Item(item_type=bridge_pb2.Item.INT, dict_key=dict_key, int_value=value)

    def deserialize_int(message, messages):
        return int(message.int_value), message

    def serialize_float(value, dict_key):
        yield bridge_pb2.Item(item_type=bridge_pb2.Item.FLOAT, dict_key=dict_key, float_value=value)

    def deserialize_float(message, messages):
        return message.float_value, message

    def serialize_str(value, dict_key):
        yield bridge_pb2.Item(item_type=bridge_pb2.Item.STR, dict_key=dict_key, str_value=value)

    def deserialize_str(message, messages):
        return str(message.str_value), message

    def serialize_ndarray(array, dict_key):
        # TODO: select more appropriate block size
        block_size = 1024 * 1024

        for block, last in BridgeSerializer.slice_ndarray(array, block_size):
            assert 0 < len(block) <= block_size
            if last:
                yield bridge_pb2.Item(
                    item_type=bridge_pb2.Item.NUMPY_ARRAY,
                    dict_key=dict_key,
                    numpy_array_value=bridge_pb2.Item.NumpyArray(
                        last=True,
                        dtype=str(array.dtype),
                        shape=array.shape,
                        data=block
                    )
                )
            else:
                yield bridge_pb2.Item(
                    item_type=bridge_pb2.Item.NUMPY_ARRAY,
                    numpy_array_value=bridge_pb2.Item.NumpyArray(
                        last=False,
                        data=block
                    )
                )

    def deserialize_ndarray(message, messages):
        data = []
        while True:
            assert message.item_type == bridge_pb2.Item.NUMPY_ARRAY
            data.append(message.numpy_array_value.data)
            if message.numpy_array_value.last:
                break
            message = next(messages)

        # optimization to avoid extra data copying if array data fits to one block
        # TODO: compare actual performance
        if len(data) == 1:
            buffer_ = data[0]
        else:
            buffer_ = ''.join(data)

        value = numpy.ndarray(
            shape=message.numpy_array_value.shape,
            dtype=numpy.dtype(message.numpy_array_value.dtype),
            buffer=buffer_
        )
        return value, message

    @staticmethod
    def slice_ndarray(array, block_size):
        assert block_size > 0

        data = array.data
        size = len(data)

        # optimization to avoid extra data copying if array data fits to one block
        # TODO: compare actual performance
        if size <= block_size:
            bytes_ = array.tobytes()
            assert size == len(bytes_)
            yield bytes_, True
        else:
            i = 0
            while True:
                ii = i + block_size
                if ii >= size:
                    break
                yield data[i:ii], False
                i = ii
            yield data[i:], True

    SERIALIZERS = {
        list:           serialize_list,
        dict:           serialize_dict,
        types.NoneType: serialize_none,
        bool:           serialize_bool,
        int:            serialize_int,
        float:          serialize_float,
        str:            serialize_str,
        numpy.ndarray:  serialize_ndarray
    }

    DESERIALIZERS = {
        bridge_pb2.Item.LIST_OPEN:   deserialize_list,
        bridge_pb2.Item.DICT_OPEN:   deserialize_dict,
        bridge_pb2.Item.NONE:        deserialize_none,
        bridge_pb2.Item.BOOL:        deserialize_bool,
        bridge_pb2.Item.INT:         deserialize_int,
        bridge_pb2.Item.FLOAT:       deserialize_float,
        bridge_pb2.Item.STR:         deserialize_str,
        bridge_pb2.Item.NUMPY_ARRAY: deserialize_ndarray
    }
