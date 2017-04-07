import pg_config

from relaax.common.algorithms import subgraph

from lib import graph
from lib import utils


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        ph_gradients = graph.Placeholders(zip(
            [pg_config.config.state_size] + pg_config.config.hidden_sizes,
            pg_config.config.hidden_sizes + [pg_config.config.action_size]
        ))
        sg_weights = graph.Variables(ph_gradients, initializer=graph.XavierInitializer())
        sg_apply_gradients = graph.ApplyGradients(
            graph.Adam(learning_rate=pg_config.config.learning_rate),
            sg_weights,
            ph_gradients
        )
        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_get_weights = sg_weights.get()
        self.op_apply_gradients = sg_apply_gradients.apply_gradients(ph_gradients)
        self.op_initialize = sg_initialize.initialize()


# Policy run by Agent(s)
class PolicyModel(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        ph_weights = graph.Placeholders(zip(
            [pg_config.config.state_size] + pg_config.config.hidden_sizes,
            pg_config.config.hidden_sizes + [pg_config.config.action_size]
        ))
        sg_weights = graph.Variables(placeholders=ph_weights)

        ph_state = graph.Placeholder((None, pg_config.config.state_size))
        ph_action = graph.Placeholder((None, pg_config.config.action_size))
        ph_discounted_reward = graph.Placeholder((None, 1))

        sg_network = graph.FullyConnected(ph_state, sg_weights)
        sg_policy_loss = graph.PolicyLoss(
            action=ph_action,
            discounted_reward=ph_discounted_reward,
            network=sg_network
        )

        sg_policy = graph.Policy(sg_network, sg_policy_loss)

        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_assign_weights = sg_weights.assign(ph_weights)
        self.op_get_action = sg_policy.get_action(ph_state)
        self.op_compute_gradients = sg_policy.compute_gradients(ph_state, ph_action, ph_discounted_reward)
        self.op_initialize = sg_initialize.initialize()


if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, PolicyModel)
