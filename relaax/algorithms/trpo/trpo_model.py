from __future__ import absolute_import
import numpy as np

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import loss
from relaax.common.algorithms.lib import utils

from . import trpo_config
from .lib import trpo_graph


class Network(subgraph.Subgraph):
    def build_graph(self):
        input = layer.Input(trpo_config.config.input)
        self.weights = layer.Weights()


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def wait_for_iteration(self, session):
        return self._ps_bridge.wait_for_iteration()

    def send_experience(self, session, n_iter, paths, length):
        self._ps_bridge.send_experience(n_iter, paths, length)

    def receive_weights(self, session, n_iter):
        return self._ps_bridge.receive_weights(n_iter)

    def build_graph(self):
        # Build graph
        
        sg_n_iter = trpo_graph.NIter()

        sg_global_step = graph.GlobalStep()
        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)
        self.op_inc_step = self.Op(sg_global_step.increment, increment=sg_global_step.ph_increment)
        self.op_initialize = self.Op(sg_initialize)

        self.call_wait_for_iteration = self.Call(self.wait_for_iteration)
        self.call_send_experience = self.Call(self.send_experience)
        self.call_receive_weights = self.Call(self.receive_weights)

        self.op_turn_collect_on = sg_n_iter.op_turn_collect_on
        self.op_turn_collect_off = sg_n_iter.op_turn_collect_off
        self.op_n_iter_value = sg_n_iter.op_n_iter_value
        self.op_n_iter = sg_n_iter.op_n_iter
        self.op_next_iter = sg_n_iter.op_next_iter


# Policy run by Agent(s)
class AgentModel(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_network = Network()

        # Expose public API


if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, AgentModel)
