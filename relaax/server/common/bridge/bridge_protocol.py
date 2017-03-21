import numpy

import bridge_pb2


class BridgeProtocol(object):

    @staticmethod
    def build_arg_messages(ops, feed_dict):
        yield bridge_pb2.ArgPart(ops=ops)
        for name, array in feed_dict.iteritems():
            for message in BridgeProtocol\
                    .build_arg_messages_for_one_array(name, array):
                yield message

    @staticmethod
    def build_arg_messages_for_one_array(name, array):
        # TODO: select more appropriate block size
        block_size = 1024 * 1024

        dtype = str(array.dtype)
        shape = array.shape
        data = array.data
        size = len(data)

        # optimization to avoid extra data
        # copying if array data fits to one block
        # TODO: compare actual performance
        if size <= block_size:
            bytes_ = array.tobytes()
            assert size == len(bytes_)
            yield bridge_pb2.ArgPart(
                dtype=dtype,
                shape=shape,
                name=name,
                last_part=True,
                data=bytes_
            )
        else:
            i = 0
            while i < size:
                ii = i + block_size
                if ii >= size:
                    yield bridge_pb2.ArgPart(
                        dtype=dtype,
                        shape=shape,
                        name=name,
                        last_part=True,
                        data=data[i:ii]
                    )
                else:
                    yield bridge_pb2.ArgPart(
                        last_part=False,
                        data=data[i:ii]
                    )
                i = ii

    @staticmethod
    def parse_arg_messages(messages):
        ops = next(messages).ops
        feed_dict = {}
        data = []
        for message in messages:
            if message.last_part:
                # optimization to avoid extra data
                # copying if array data fits to one block
                # TODO: compare actual performance
                if len(data) == 0:
                    feed_dict[message.name] = numpy.ndarray(
                        shape=message.shape,
                        dtype=numpy.dtype(message.dtype),
                        buffer=message.data
                    )
                else:
                    data.append(message.data)
                    feed_dict[message.name] = numpy.ndarray(
                        shape=message.shape,
                        dtype=numpy.dtype(message.dtype),
                        buffer=''.join(data)
                    )
                    data = []
            else:
                data.append(message.data)
        assert len(data) == 0
        return list(ops), feed_dict

    @staticmethod
    def build_result_messages(arrays):
        for array in arrays:
            for message in BridgeProtocol\
                    .build_result_messages_for_one_array(array):
                yield message

    @staticmethod
    def build_result_messages_for_one_array(array):
        # TODO: select more appropriate block size
        block_size = 1024 * 1024

        dtype = str(array.dtype)
        shape = array.shape
        data = array.data
        size = len(data)

        # optimization to avoid extra data
        # copying if array data fits to one block
        # TODO: compare actual performance
        if size <= block_size:
            bytes_ = array.tobytes()
            assert size == len(bytes_)
            yield bridge_pb2.ResultPart(
                dtype=dtype,
                shape=shape,
                last_part=True,
                data=bytes_
            )
        else:
            i = 0
            while i < size:
                ii = i + block_size
                if ii >= size:
                    yield bridge_pb2.ResultPart(
                        dtype=dtype,
                        shape=shape,
                        last_part=True,
                        data=data[i:ii]
                    )
                else:
                    yield bridge_pb2.ResultPart(
                        last_part=False,
                        data=data[i:ii]
                    )
                i = ii

    @staticmethod
    def parse_result_messages(messages):
        data = []
        for message in messages:
            if message.last_part:
                # optimization to avoid extra data
                # copying if array data fits to one block
                # TODO: compare actual performance
                if len(data) == 0:
                    yield numpy.ndarray(
                        shape=message.shape,
                        dtype=numpy.dtype(message.dtype),
                        buffer=message.data
                    )
                else:
                    data.append(message.data)
                    yield numpy.ndarray(
                        shape=message.shape,
                        dtype=numpy.dtype(message.dtype),
                        buffer=''.join(data)
                    )
                    data = []
            else:
                data.append(message.data)
