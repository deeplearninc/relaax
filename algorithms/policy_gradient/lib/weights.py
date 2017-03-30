import tensorflow as tf

from relaax.common.algorithms.decorators import SubGraph


class Weights(SubGraph):
    """Holder for variables representing weights of the fully connected NN."""

    @staticmethod
    def build(self, shapes, initializer):
        """Assemble weights of the NN into tf graph.

        Args:
            shapes: sizes of the weights variables
            initializer: initializer for variables

        Returns:
            list to the 'weights' tensors in the graph

        """
        return [
            tf.Variable(shape=shape, initializer=initializer.tf_initializer())
            for shape in shapes
        ]
