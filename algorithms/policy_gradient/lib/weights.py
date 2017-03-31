import tensorflow as tf
import numpy as np

from initializers import Zero
from relaax.common.algorithms.subgraph import Subgraph
from pg_config import config


class Weights(Subgraph):
    """Holder for variables representing weights of the fully connected NN."""

    def build(self, initializer=None):
        """Assemble weights of the NN into tf graph.

        Args:
            shapes: sizes of the weights variables
            initializer: initializer for variables

        Returns:
            list to the 'weights' tensors in the graph

        """

        if initializer is None:
            initializer = Zero()

        state_size=config.state_size
        hidden_sizes=config.hidden_layers
        action_size=config.action_size

        shapes = zip([state_size] + hidden_sizes, hidden_sizes + [action_size])
        return [
            tf.Variable(initial_value=initializer(shape=shape, dtype=np.float32))
            for shape in shapes
        ]
