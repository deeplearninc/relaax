from __future__ import absolute_import
import numpy as np

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import loss
from relaax.common.algorithms.lib import utils

from . import trpo_config

from .old_trpo_gae.parameter_server import parameter_server


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
        self._ps_bridge = parameter_server.ParameterServer(trpo_config.config, None, None).bridge()

        # Build graph
        sg_global_step = graph.GlobalStep()
        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)
        self.op_initialize = self.Op(sg_initialize)

        self.call_wait_for_iteration = self.Call(self.wait_for_iteration)
        self.call_send_experience = self.Call(self.send_experience)
        self.call_receive_weights = self.Call(self.receive_weights)


# Policy run by Agent(s)
class AgentModel(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_network = Network()

        # Expose public API


if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, AgentModel)
