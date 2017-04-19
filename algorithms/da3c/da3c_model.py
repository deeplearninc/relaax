
from relaax.common.algorithms import subgraph

from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import utils


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_initialize = sg_initialize.initialize()


# Policy run by Agent(s)
class AgentModel(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_initialize = sg_initialize.initialize()


if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, PolicyModel)
