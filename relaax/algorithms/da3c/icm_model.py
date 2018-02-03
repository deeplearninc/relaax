from __future__ import absolute_import
import numpy as np

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from . import da3c_config as cfg


class ICM(subgraph.Subgraph):
    def build_graph(self, input):
        shape = [None] + [cfg.config.output.action_size]
        self.ph_probs = graph.Placeholder(np.float32, shape=shape, name='act_probs')
        self.ph_taken = graph.Placeholder(np.int32, shape=(None,), name='act_taken')

        flattened_input = layer.Flatten(input)
        last_size = flattened_input.node.shape.as_list()[-1]

        inverse_inp = graph.Reshape(input, [-1, last_size*2])

        get_first = graph.TfNode(inverse_inp.node[:, :last_size])
        get_second = graph.TfNode(inverse_inp.node[:, last_size:])

        forward_inp = graph.Concat([get_first, self.ph_probs], axis=1)

        fc_size = cfg.config.hidden_sizes[-1]
        std_init = 1e-2

        inv_fc1 = layer.Dense(inverse_inp, fc_size, layer.Activation.Relu, init_var=std_init)
        inv_fc2 = layer.Dense(inv_fc1, shape[-1], init_var=std_init)   # layer.Activation.Softmax

        fwd_fc1 = layer.Dense(forward_inp, fc_size, layer.Activation.Relu, init_var=std_init)
        fwd_fc2 = layer.Dense(fwd_fc1, last_size, init_var=std_init)

        inv_loss = graph.SparseSoftmaxCrossEntropyWithLogits(inv_fc2, self.ph_taken).op
        fwd_loss = graph.L2loss(fwd_fc2.node - get_second.node).op

        self.ph_state = input.ph_state  # should be even wrt to batch_size for now
        self.rew_out = graph.TfNode(cfg.config.icm.nu * fwd_loss)

        self.loss = graph.TfNode(10*(cfg.config.icm.beta * fwd_loss + (1 - cfg.config.icm.beta) * inv_loss))

        layers = [input, inv_fc1, inv_fc2, fwd_fc1, fwd_fc2]
        self.weights = layer.Weights(*layers)
