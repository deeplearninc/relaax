import tensorflow as tf
import numpy as np

from relaax.common.algorithms.subgraph import Subgraph


class Weights(Subgraph):
    """Holder for variables representing weights of the fully connected NN."""

    def assemble(self, shapes, initializer=None):
        """Assemble weights of the NN into tf graph.

        Args:
            shapes: sizes of the weights variables
            initializer: initializer for variables

        Returns:
            list to the 'weights' tensors in the graph

        """
        return [tf.Variable(initial_value=np.zeros(shape=shape))
                for shape in shapes]
