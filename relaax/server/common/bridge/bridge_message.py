import numpy
import types

import bridge_pb2


class BridgeMessage(object):

    @staticmethod
    def serialize(value):
        return BridgeMessage.serialize_recursive(value, dict_key=None)

    @staticmethod
    def deserialize(messages):
        value, _ = BridgeMessage.deserialize_recursive(next(messages), messages)
        the_end = object()
        assert next(messages, the_end) == the_end
        return value

    @staticmethod
    def serialize_recursive(value, dict_key):
        return BridgeMessage.SERIALIZERS[type(value)](value, dict_key)

    @staticmethod
    def deserialize_recursive(message, messages):
        return BridgeMessage.DESERIALIZERS[message.item_type](message, messages)

    class NoneMarshal(object):
        def __init__(self):
            self.type = types.NoneType
            self.item_type = bridge_pb2.Item.NONE

        def serialize(self, value, dict_key):
            yield bridge_pb2.Item(item_type=self.item_type, dict_key=dict_key)

        def deserialize(self, message, messages):
            return None, message

    class ScalarMarshal(object):
        def __init__(self, item_type, type, value_attr):
            self.item_type = item_type
            self.type = type
            self.value_attr = value_attr

        def serialize(self, value, dict_key):
            item = bridge_pb2.Item(item_type=self.item_type, dict_key=dict_key)
            setattr(item, self.value_attr, value)
            yield item

        def deserialize(self, message, messages):
            return self.type(getattr(message, self.value_attr)), message

    class NdarrayMarshal(object):
        def __init__(self):
            self.type = numpy.ndarray
            self.item_type = bridge_pb2.Item.NUMPY_ARRAY

        def serialize(self, array, dict_key):
            # TODO: select more appropriate block size
            block_size = 1024 * 1024

            for block, last in self.slice_ndarray(array, block_size):
                assert 0 < len(block) <= block_size
                if last:
                    yield bridge_pb2.Item(
                        item_type=self.item_type,
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
                        item_type=self.item_type,
                        numpy_array_value=bridge_pb2.Item.NumpyArray(
                            last=False,
                            data=block
                        )
                    )

        def deserialize(self, message, messages):
            data = []
            while True:
                assert message.item_type == self.item_type
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

        def slice_ndarray(self, array, block_size):
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
                for i in xrange(0, size, block_size):
                    yield data[i:i + block_size], i + block_size >= size

    class ListMarshal(object):
        def __init__(self):
            self.type = list
            self.item_type = bridge_pb2.Item.LIST_OPEN

        def serialize(self, value, dict_key):
            yield bridge_pb2.Item(item_type=self.item_type)
            for item in value:
                for message in BridgeMessage.serialize_recursive(item, dict_key=None):
                    yield message
            yield bridge_pb2.Item(item_type=bridge_pb2.Item.LIST_CLOSE, dict_key=dict_key)

        def deserialize(self, message, messages):
            value = []
            while True:
                message = next(messages)
                if message.item_type == bridge_pb2.Item.LIST_CLOSE:
                    return value, message
                value.append(BridgeMessage.deserialize_recursive(message, messages)[0])

    class DictMarshal(object):
        def __init__(self):
            self.type = dict
            self.item_type = bridge_pb2.Item.DICT_OPEN

        def serialize(self, value, dict_key):
            yield bridge_pb2.Item(item_type=self.item_type)
            for key, item in value.iteritems():
                for message in BridgeMessage.serialize_recursive(item, dict_key=key):
                    yield message
            yield bridge_pb2.Item(item_type=bridge_pb2.Item.DICT_CLOSE, dict_key=dict_key)

        def deserialize(self, message, messages):
            value = {}
            while True:
                message = next(messages)
                if message.item_type == bridge_pb2.Item.DICT_CLOSE:
                    return value, message
                item, last_message = BridgeMessage.deserialize_recursive(message, messages)
                value[last_message.dict_key] = item

    SERIALIZERS = {}
    DESERIALIZERS = {}

    for converter in [
        NoneMarshal(),
        ScalarMarshal(bridge_pb2.Item.BOOL, bool, 'bool_value'),
        ScalarMarshal(bridge_pb2.Item.INT, int, 'int_value'),
        ScalarMarshal(bridge_pb2.Item.NUMPY_INT_32, numpy.int32, 'int_value'),
        ScalarMarshal(bridge_pb2.Item.FLOAT, float, 'float_value'),
        ScalarMarshal(bridge_pb2.Item.STR, str, 'str_value'),
        NdarrayMarshal(),
        ListMarshal(),
        DictMarshal()
    ]:
        assert converter.type not in SERIALIZERS
        SERIALIZERS[converter.type] = converter.serialize
        assert converter.item_type not in DESERIALIZERS
        DESERIALIZERS[converter.item_type] = converter.deserialize
