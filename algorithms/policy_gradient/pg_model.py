from pg_config import config

from relaax.common.algorithms.subgraph import Subgraph

from lib.graph import Variables, Policy, PolicyLoss, FullyConnected, Placeholder, Placeholders
from lib.graph import ApplyGradients, Adam, XavierInitializer, Initialize
from lib.utils import assemble_and_show_graphs


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(Subgraph):
    def build_graph(self):
        # Build graph
        ph_gradients = Placeholders(zip(
            [config.state_size] + config.hidden_sizes,
            config.hidden_sizes + [config.action_size]
        ))
        sg_weights = Variables(ph_gradients, initializer=XavierInitializer())
        sg_apply_gradients = ApplyGradients(
            Adam(learning_rate=config.learning_rate),
            sg_weights,
            ph_gradients
        )
        sg_initialize = Initialize()

        # Expose public API
        self.op_get_weights = sg_weights.get()
        self.op_apply_gradients = sg_apply_gradients.apply_gradients(ph_gradients)
        self.op_initialize = sg_initialize.initialize()


# Policy run by Agent(s)
class PolicyModel(Subgraph):
    def build_graph(self):
        # Build graph
        ph_weights = Placeholders(zip(
            [config.state_size] + config.hidden_sizes,
            config.hidden_sizes + [config.action_size]
        ))
        sg_weights = Variables(placeholders=ph_weights)

        ph_state = Placeholder((None, config.state_size))
        ph_action = Placeholder((None, config.action_size))
        ph_discounted_reward = Placeholder((None, 1))

        sg_network = FullyConnected(ph_state, sg_weights)
        sg_policy_loss = PolicyLoss(
            action=ph_action,
            discounted_reward=ph_discounted_reward,
            network=sg_network
        )

        sg_policy = Policy(sg_network, sg_policy_loss)

        sg_initialize = Initialize()

        # Expose public API
        self.op_assign_weights = sg_weights.assign(ph_weights)
        self.op_get_action = sg_policy.get_action(ph_state)
        self.op_compute_gradients = sg_policy.compute_gradients(ph_state, ph_action, ph_discounted_reward)
        self.op_initialize = sg_initialize.initialize()


if __name__ == '__main__':
    assemble_and_show_graphs(SharedParameters, PolicyModel)
