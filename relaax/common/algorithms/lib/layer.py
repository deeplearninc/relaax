from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph


class Convolution(subgraph.Subgraph):
    def build_graph(self, x, n_filters, filter_size, stride,
            border=graph.ValidBorder, activation=graph.Relu):

        self.weight = graph.Wb(np.float32, (*filter_size, 4, n_filters))

        return activation(tf.nn.conv2d(x, self.weight.W,
                strides=[1] + stride + [1], padding=border.padding) +
                self.weight.b)


        self.conv1 = graph.Wb(np.float32, (8, 8, 4, 16)) # stride=4

        conv1 = layer.Convolution2D(state, 16, 8, 8, subsample=(4, 4),
                border_mode='valid', activation=activation.Relu)

        return tf.nn.conv2d(x.node, wb.W, strides=[1, stride, stride, 1], padding="VALID") + wb.b

        assert len(x.shape) == 1
        self.weight = graph.Wb(np.float32, (x.shape[0], size))
        return activation(graph.ApplyWb(x, self.weight))


class Dense(subgraph.Subgraph):
    def build_graph(self, x, size, activation=graph.Relu):
        assert len(x.shape) == 1
        self.weight = graph.Wb(np.float32, (x.shape[0], size))
        return activation(graph.ApplyWb(x, self.weight))
