import numpy as np

from relaax.common.algorithms import subgraph
from lib import graph
from relaax.common.algorithms.lib import utils
import da3c_config


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_weights = graph.Weights()
        ph_gradients = graph.Placeholders(sg_weights)
        ph_n_steps = graph.Placeholder(np.int64)
        sg_n_step = graph.Counter(ph_n_steps, np.int64)
        sg_apply_gradients = graph.ApplyGradients(sg_weights, ph_gradients, sg_n_step)
        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_n_step = sg_n_step.value()
        self.op_get_weights = sg_weights.get()
        self.op_apply_gradients = sg_apply_gradients.apply_gradients(ph_gradients, ph_n_steps)
        self.op_initialize = sg_initialize.initialize()


# Policy run by Agent(s)
class AgentModel(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_weights = graph.Weights()
        ph_weights = graph.Placeholders(sg_weights)
        sg_assign_weights = graph.Assign(sg_weights, ph_weights)
        ph_state = graph.Placeholder(np.float32, shape=
            [None] +
            da3c_config.options.get('algorithm/state_shape') +
            [da3c_config.options.get('algorithm/history_len')]
        )
        ph_action = graph.Placeholder(np.int32, shape=(None, ))
        ph_value = graph.Placeholder(np.float32, shape=(None, ))
        ph_discounted_reward = graph.Placeholder(np.float32, shape=(None, ))
        sg_network = graph.Network(ph_state, sg_weights)
        sg_loss = graph.Loss(ph_state, ph_action, ph_value, ph_discounted_reward, sg_weights, sg_network)
        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_assign_weights = sg_assign_weights.assign(ph_weights)
        self.op_get_action_and_value = sg_network.get_action_and_value(ph_state)
        self.op_compute_gradients = sg_loss.compute_gradients(ph_state, ph_action, ph_value, ph_discounted_reward)
        self.op_initialize = sg_initialize.initialize()


if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, AgentModel)
