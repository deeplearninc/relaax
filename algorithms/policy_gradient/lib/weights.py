import tensorflow as tf


class Weights(object):
    """Holder for variables representing weights of the fully connected NN."""

    @classmethod
    def assemble(cls, input_size, output_size, hidden_layers):
        """Assemble weights of the NN into tf graph.

        Args:
            input_size: input layer size
            output_size: output layer size
            hidden_layers: sizes of the hidden layers

        Returns:
            list to the 'weights' tensors in the graph

        """
        weights = []

        def add_layer(name, input_size, layer_size):
            return weights.append(
                tf.get_variable(name, shape=[input_size, layer_size]))

        # input layer weights
        add_layer('W0', input_size, hidden_layers[0])

        # hidden layer weights
        nlayers = len(hidden_layers)
        for i in range(0, nlayers - 1):
            add_layer(
                'W%d' % (i + 1), hidden_layers[i], hidden_layers[i + 1])

        # output layer weights
        add_layer('W%d' % nlayers, hidden_layers[-1], output_size)

        return weights
