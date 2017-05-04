import numpy as np

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import utils

import pg_config


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def build_graph(self):
        shapes = zip(
            [pg_config.config.state_size] + pg_config.config.hidden_sizes,
            pg_config.config.hidden_sizes + [pg_config.config.action_size]
        )

        # Build graph
        ph_n_steps = graph.Placeholder(np.int64)
        sg_n_step = graph.Counter(ph_n_steps, np.int64)
        sg_weights = graph.List(graph.Wb(np.float32, shape) for shape in shapes)
        ph_gradients = graph.PlaceholdersByVariables(sg_weights)
        sg_apply_gradients = graph.ApplyGradients(
            graph.AdamOptimizer(learning_rate=pg_config.config.learning_rate),
            sg_weights,
            ph_gradients
        )
        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_n_step = sg_n_step.value()
        self.op_get_weights = sg_weights.get()
        self.op_apply_gradients = sg_apply_gradients.apply_gradients(ph_gradients)
        self.op_initialize = sg_initialize.initialize()


# Policy run by Agent(s)
class PolicyModel(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        shapes = zip(
            [pg_config.config.state_size] + pg_config.config.hidden_sizes,
            pg_config.config.hidden_sizes + [pg_config.config.action_size]
        )

        sg_weights = graph.List(graph.Wb(np.float32, shape) for shape in shapes)
        ph_weights = graph.PlaceholdersByVariables(sg_weights)

        ph_state = graph.Placeholder(np.float32, (None, pg_config.config.state_size))
        ph_action = graph.Placeholder(np.int32, (None, ))
        ph_discounted_reward = graph.Placeholder(np.float32, (None, 1))

        sg_network = graph.FullyConnected(ph_state, sg_weights)
        sg_policy_loss = graph.PolicyLoss(
            action=ph_action,
            action_size=pg_config.config.action_size,
            discounted_reward=ph_discounted_reward,
            network=sg_network
        )

        sg_policy = graph.Policy(sg_network, sg_policy_loss)

        sg_assign_weights = graph.Assign(sg_weights, ph_weights)

        # Expose public API
        self.op_assign_weights = sg_assign_weights.assign(ph_weights)
        self.op_get_action = sg_policy.get_action(ph_state)
        self.op_compute_gradients = sg_policy.compute_gradients(ph_state, ph_action, ph_discounted_reward)


if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, PolicyModel)
