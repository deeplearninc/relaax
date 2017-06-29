from __future__ import absolute_import
import numpy as np
import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import utils


class SharedParameters(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_global_step = graph.GlobalStep()
        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)
        self.op_initialize = self.Op(sg_initialize)


class Model(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        state = graph.Placeholder(np.float32, shape=(2, ))
        reverse = graph.TfNode(tf.reverse(state.node, [0]))

        # Expose public API
        self.op_get_action = self.Op(reverse, state=state)

if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, Model)
