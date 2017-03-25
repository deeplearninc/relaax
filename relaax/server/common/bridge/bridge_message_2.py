import numpy
import types
import bridge_pb2


class BridgeMessage2(object):
    @classmethod
    def serialize(cls, value):
        return cls.INSTANCE.serialize_any(value, dict_key=None)

    @classmethod
    def deserialize(cls, messages):
        value, _ = cls.INSTANCE.deserialize_any(next(messages), messages)
        the_end = object()
        assert next(messages, the_end) == the_end
        return value

    def __init__(self):
        self.serializers = {}
        self.deserializers = {}

        for item_type, value_type, serialize, deserialize in [
            self.none_marshaller(bridge_pb2.Item.NONE, types.NoneType),
            self.scalar_marshaller(bridge_pb2.Item.BOOL, bool, 'bool_value'),
            self.scalar_marshaller(bridge_pb2.Item.INT, int, 'int_value'),
            self.scalar_marshaller(bridge_pb2.Item.NUMPY_INT_32, numpy.int32, 'int_value'),
            self.scalar_marshaller(bridge_pb2.Item.FLOAT, float, 'float_value'),
            self.scalar_marshaller(bridge_pb2.Item.STR, str, 'str_value'),
            self.ndarray_marshaller(bridge_pb2.Item.NUMPY_ARRAY, numpy.ndarray),
            self.list_marshaller(bridge_pb2.Item.LIST_OPEN, list),
            self.dict_marshaller(bridge_pb2.Item.DICT_OPEN, dict)
        ]:
            assert item_type not in self.deserializers
            self.deserializers[item_type] = deserialize
            assert value_type not in self.serializers
            self.serializers[value_type] = serialize

    def serialize_any(self, value, dict_key):
        return self.serializers[type(value)](value, dict_key)

    def deserialize_any(self, message, messages):
        return self.deserializers[message.item_type](message, messages)

    def none_marshaller(self, item_type, value_type):
        def serialize(value, dict_key):
            yield bridge_pb2.Item(item_type=item_type, dict_key=dict_key)

        def deserialize(message, messages):
            return None, message

        return item_type, value_type, serialize, deserialize

    def scalar_marshaller(self, item_type, value_type, attr_name):
        def serialize(value, dict_key):
            item = bridge_pb2.Item(item_type=item_type, dict_key=dict_key)
            setattr(item, attr_name, value)
            yield item

        def deserialize(message, messages):
            return value_type(getattr(message, attr_name)), message

        return item_type, value_type, serialize, deserialize

    def list_marshaller(self, item_type, value_type):
        def serialize(value, dict_key):
            yield bridge_pb2.Item(item_type=item_type)
            for item in value:
                for message in self.serialize_any(item, dict_key=None):
                    yield message
            yield bridge_pb2.Item(item_type=bridge_pb2.Item.LIST_CLOSE, dict_key=dict_key)

        def deserialize(message, messages):
            value = []
            while True:
                message = next(messages)
                if message.item_type == bridge_pb2.Item.LIST_CLOSE:
                    return value, message
                value.append(self.deserialize_any(message, messages)[0])

        return item_type, value_type, serialize, deserialize

    def dict_marshaller(self, item_type, value_type):
        def serialize(value, dict_key):
            yield bridge_pb2.Item(item_type=item_type)
            for key, item in value.iteritems():
                for message in self.serialize_any(item, dict_key=key):
                    yield message
            yield bridge_pb2.Item(item_type=bridge_pb2.Item.DICT_CLOSE, dict_key=dict_key)

        def deserialize(message, messages):
            value = {}
            while True:
                message = next(messages)
                if message.item_type == bridge_pb2.Item.DICT_CLOSE:
                    return value, message
                item, last_message = self.deserialize_any(message, messages)
                value[last_message.dict_key] = item

        return item_type, value_type, serialize, deserialize

    def ndarray_marshaller(self, item_type, value_type):
        def serialize(array, dict_key):
            # TODO: select more appropriate block size
            block_size = 1024 * 1024

            for block, last in slice_ndarray(array, block_size):
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

        def deserialize(message, messages):
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
                for i in xrange(0, size, block_size):
                    yield data[i:i + block_size], i + block_size >= size

        return item_type, value_type, serialize, deserialize


BridgeMessage2.INSTANCE = BridgeMessage2()
