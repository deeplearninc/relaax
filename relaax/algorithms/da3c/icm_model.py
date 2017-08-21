from __future__ import absolute_import
import numpy as np

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from . import da3c_config as cfg


class ICM(subgraph.Subgraph):
    def build_graph(self):
        conv_layer = dict(type=layer.Convolution, activation=layer.Activation.Elu,
                          n_filters=32, filter_size=[3, 3], stride=[2, 2],
                          border=layer.Border.Same)
        input = layer.Input(cfg.config.input, descs=[dict(conv_layer)] * 4)

        shape = [None] + [cfg.config.output.action_size]
        self.ph_action = graph.Placeholder(np.float32, shape=shape, name='pred_action')

        flattened_input = layer.Flatten(input)
        last_size = flattened_input.node.shape.as_list()[-1]

        inverse_inp = graph.Reshape(input, [-1, last_size*2])

        get_first = graph.TfNode(inverse_inp.node[:, :last_size])
        get_second = graph.TfNode(inverse_inp.node[:, last_size:])

        forward_inp = graph.Concat([get_first, self.ph_action], axis=1)

        fc_size = cfg.config.hidden_sizes[-1]
        inv_fc1 = layer.Dense(inverse_inp, fc_size, layer.Activation.Relu)
        inv_fc2 = layer.Dense(inv_fc1, shape[-1], layer.Activation.Softmax)

        fwd_fc1 = layer.Dense(forward_inp, fc_size, layer.Activation.Relu)
        fwd_fc2 = layer.Dense(fwd_fc1, last_size)

        self.ph_state = input.ph_state  # should be even wrt to batch_size for now
        self.inv_out = inv_fc2

        self.discrepancy = graph.L2loss(fwd_fc2.node - get_second.node)
        self.rew_out = cfg.config.icm.nu * self.discrepancy.node

        layers = [input, inv_fc1, inv_fc2, fwd_fc1, fwd_fc2]
        self.weights = layer.Weights(*layers)
