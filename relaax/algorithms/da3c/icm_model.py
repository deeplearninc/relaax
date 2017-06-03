from __future__ import absolute_import
import numpy as np

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import loss
from relaax.common.algorithms.lib import utils
from . import da3c_config


class ICM(subgraph.Subgraph):
    def build_graph(self):
        conv_layer = dict(type=layer.Convolution, activation=layer.Activation.Elu,
                          n_filters=32, filter_size=[3, 3], stride=[1, 1],
                          border=layer.Border.Same)
        input = layer.Input(da3c_config.config.input, descs=[dict(conv_layer)] * 4)

        shape = [None] + [da3c_config.config.output.action_size]
        self.ph_action = graph.Placeholder(np.float32, shape=shape, name='pred_action')

        print('input:', input.node.get_shape())
        inverse = graph.Reshape(input, [-1, 288*2])
        print('inverse:', inverse.node.get_shape())

        get_first = inverse.node[:, :288]
        get_second = inverse.node[:, 288:]
        print('get_first:', get_first.get_shape())
        forward = graph.Concat(1, [get_first, self.ph_action])
        print('forward:', forward.node.get_shape())

        inv_fc1 = layer.Dense(inverse, 256)  # , layer.Activation.Elu
        inv_fc2 = layer.Dense(inv_fc1, 4)    # , layer.Activation.Softmax | Actor

        fwd_fc1 = layer.Dense(forward, 256)  # , layer.Activation.Elu
        fwd_fc2 = layer.Dense(fwd_fc1, 288)  # , layer.Activation.Elu

        self.ph_state = input.ph_state
        self.inv_out = inv_fc2

        nu = graph.Constant(0.8)  # scaling factor, which needs to retrieve from the config
        self.discrepancy = graph.L2loss(fwd_fc2.node - get_second.node)
        self.rew_out = nu.node * self.discrepancy.node

        layers = [input, inv_fc1, inv_fc2, fwd_fc1, fwd_fc2]
        self.weights = layer.Weights(*layers)
