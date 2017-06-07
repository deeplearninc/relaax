from __future__ import absolute_import
from builtins import next
from builtins import str
from builtins import range
from builtins import object
import numpy
import types
from . import bridge_pb2


class MessageStream(object):
    def __init__(self, messages):
        self.messages = messages
        self.first = next(messages)

    def __next__(self):
        self.first = next(self.messages)
        return self.first


class BaseMarshaller(object):
    def __init__(self, item_type, value_type):
        self.item_type = item_type
        self.value_type = value_type


class NoneMarshaller(BaseMarshaller):
    def serialize(self, value, dict_key):
        yield bridge_pb2.Item(item_type=self.item_type, dict_key=dict_key)

    def deserialize(self, stream):
        return None


class ScalarMarshaller(BaseMarshaller):
    def __init__(self, item_type, value_type, value_attr):
        super(ScalarMarshaller, self).__init__(item_type, value_type)
        self.value_attr = value_attr

    def serialize(self, value, dict_key):
        item = bridge_pb2.Item(item_type=self.item_type, dict_key=dict_key)
        setattr(item, self.value_attr, value)
        yield item

    def deserialize(self, stream):
        return self.value_type(getattr(stream.first, self.value_attr))


class NdarrayMarshaller(BaseMarshaller):
    def serialize(self, array, dict_key):
        # TODO: select more appropriate block size
        block_size = 1024 * 1024

        for block, last in self.slice_ndarray(array, block_size):
            assert 0 <= len(block) <= block_size
            yield bridge_pb2.Item(
                item_type=self.item_type,
                dict_key=dict_key if last else None,
                numpy_array_value=bridge_pb2.Item.NumpyArray(
                    last=last,
                    dtype=str(array.dtype) if last else None,
                    shape=array.shape if last else None,
                    data=block
                )
            )

    def deserialize(self, stream):
        data = []
        while True:
            assert stream.first.item_type == self.item_type
            data.append(stream.first.numpy_array_value.data)
            if stream.first.numpy_array_value.last:
                break
            next(stream)

        value = numpy.ndarray(
            shape=stream.first.numpy_array_value.shape,
            dtype=numpy.dtype(stream.first.numpy_array_value.dtype),
            # optimization to avoid extra data copying if array data fits to one block
            # TODO: compare actual performance
            buffer=data[0] if len(data) == 1 else b''.join(data)
        )
        return value

    def slice_ndarray(self, array, block_size):
        assert block_size > 0

        bytes = array.tobytes()
        size = array.nbytes

        # optimization to avoid extra data copying if array data fits to one block
        # TODO: compare actual performance
        if size <= block_size:
            yield bytes, True
        else:
            for i in range(0, size, block_size):
                yield bytes[i:i + block_size], i + block_size >= size


class ContainerMarshaller(BaseMarshaller):
    def __init__(self, item_type, value_type, end_item_type):
        super(ContainerMarshaller, self).__init__(item_type, value_type)
        self.end_item_type = end_item_type

    def serialize(self, value, dict_key):
        yield bridge_pb2.Item(item_type=self.item_type)
        for key, item in self.items(value):
            for message in BridgeMessage.serialize_any(item, dict_key=key):
                yield message
        yield bridge_pb2.Item(item_type=self.end_item_type, dict_key=dict_key)

    def deserialize(self, stream):
        container = self.new_container()
        while next(stream).item_type != self.end_item_type:
            self.insert_item(container, BridgeMessage.deserialize_any(stream), stream.first)
        return self.cast(container)

    def new_container(self):
        return self.value_type()

    def cast(self, container):
        return container


class ListMarshaller(ContainerMarshaller):
    def items(self, container):
        return ((None, item) for item in container)

    def insert_item(self, container, item, _):
        container.append(item)


class TupleMarshaller(ListMarshaller):
    def new_container(self):
        return []

    def cast(self, container):
        return self.value_type(container)


class DictMarshaller(ContainerMarshaller):
    def items(self, container):
        return iter(container.items())

    def insert_item(self, container, item, last_message):
        container[last_message.dict_key] = item


class BridgeMessage(object):
    @classmethod
    def serialize(cls, value):
        return cls.serialize_any(value, dict_key=None)

    @classmethod
    def deserialize(cls, messages):
        value = cls.deserialize_any(MessageStream(messages))
        the_end = object()
        assert next(messages, the_end) == the_end
        return value

    @classmethod
    def serialize_any(cls, value, dict_key):
        return cls.SERIALIZERS[type(value)](value, dict_key)

    @classmethod
    def deserialize_any(cls, stream):
        return cls.DESERIALIZERS[stream.first.item_type](stream)

    @classmethod
    def initialize(cls):
        cls.SERIALIZERS = {}
        cls.DESERIALIZERS = {}

        for marshaller in [
            NoneMarshaller(bridge_pb2.Item.NONE, type(None)),
            ScalarMarshaller(bridge_pb2.Item.BOOL, bool, 'bool_value'),
            ScalarMarshaller(bridge_pb2.Item.INT, int, 'int_value'),
            ScalarMarshaller(bridge_pb2.Item.NUMPY_INT_32, numpy.int32, 'int_value'),
            ScalarMarshaller(bridge_pb2.Item.NUMPY_INT_64, numpy.int64, 'int_value'),
            ScalarMarshaller(bridge_pb2.Item.FLOAT, float, 'float_value'),
            ScalarMarshaller(bridge_pb2.Item.STR, type(''), 'str_value'),
            NdarrayMarshaller(bridge_pb2.Item.NUMPY_ARRAY, numpy.ndarray),
            ListMarshaller(bridge_pb2.Item.LIST_OPEN, list, bridge_pb2.Item.LIST_CLOSE),
            TupleMarshaller(bridge_pb2.Item.TUPLE_OPEN, tuple, bridge_pb2.Item.TUPLE_CLOSE),
            DictMarshaller(bridge_pb2.Item.DICT_OPEN, dict, bridge_pb2.Item.DICT_CLOSE)
        ]:
            assert marshaller.value_type not in cls.SERIALIZERS
            cls.SERIALIZERS[marshaller.value_type] = marshaller.serialize
            assert marshaller.item_type not in cls.DESERIALIZERS
            cls.DESERIALIZERS[marshaller.item_type] = marshaller.deserialize


BridgeMessage.initialize()
