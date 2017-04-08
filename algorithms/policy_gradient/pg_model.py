from relaax.common.algorithms import subgraph

from lib import graph
from lib import utils


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def build_graph(self):
        sg_initialize = graph.Initialize()
        self.op_initialize = sg_initialize.initialize()


# Policy run by Agent(s)
class PolicyModel(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        self.build_ps_graph()

        ph_weights = graph.Placeholders([[4, 2]])
        sg_weights = graph.Variables(placeholders=ph_weights)

        ph_state = graph.Placeholder((None, 4))
        ph_action = graph.Placeholder((None, 2))
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
        self.op_log_like = sg_policy_loss.log_like(ph_state, ph_action)
        self.op_production = sg_policy_loss.production(ph_state, ph_action, ph_discounted_reward)

    def build_ps_graph(self):
        # Build graph
        ph_gradients = graph.Placeholders([[4, 2]])
        sg_weights = graph.Variables(ph_gradients)
        sg_apply_gradients = graph.ApplyGradients(
            graph.AdamOptimizer(learning_rate=0.01),
            sg_weights,
            ph_gradients
        )
        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_get_weights = sg_weights.get()
        self.op_apply_gradients = sg_apply_gradients.apply_gradients(ph_gradients)


if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, PolicyModel)
