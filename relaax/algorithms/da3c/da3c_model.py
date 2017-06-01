from __future__ import absolute_import

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import loss
from relaax.common.algorithms.lib import utils
from .lib import da3c_graph
from . import da3c_config


class Network(subgraph.Subgraph):
    def build_graph(self):
        input = layer.Input(da3c_config.config.input)
        sizes = da3c_config.config.hidden_sizes

        sizes = da3c_config.config.hidden_sizes
        dense = layer.GenericLayers(layer.Flatten(input),
                [dict(type=layer.Dense, size=size, activation=layer.Activation.Relu)
                for size in sizes])
        head = dense
        if da3c_config.config.use_lstm:
            lstm = layer.LSTM(graph.Expand(dense, 0), size=sizes[-1])
            head = graph.Reshape(lstm, [-1, sizes[-1]])

        actor = layer.Actor(head, da3c_config.config.output)
        critic = layer.Dense(head, 1)

        self.ph_state = input.ph_state
        if da3c_config.config.use_lstm:
            self.ph_lstm_state = lstm.ph_state
            self.ph_lstm_step = lstm.ph_step
            self.lstm_zero_state = lstm.zero_state
            self.lstm_state = lstm.state
        self.actor = actor
        self.critic = graph.Flatten(critic)
        layers = [input, dense, actor, critic]
        if da3c_config.config.use_lstm:
            layers.append(lstm)
        self.weights = layer.Weights(*layers)


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_global_step = graph.GlobalStep()
        sg_weights = Network().weights
        sg_learning_rate = da3c_graph.LearningRate(sg_global_step)
        sg_optimizer = graph.RMSPropOptimizer(
            learning_rate=sg_learning_rate,
            decay=da3c_config.config.RMSProp.decay,
            momentum=0.0,
            epsilon=da3c_config.config.RMSProp.epsilon
        )
        sg_gradients = layer.Gradients(sg_weights, optimizer=sg_optimizer)
        sg_initialize = graph.Initialize()

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)
        self.op_get_weights = self.Op(sg_weights)
        self.op_apply_gradients = self.Ops(sg_gradients.apply,
                sg_global_step.increment, gradients=sg_gradients.ph_gradients,
                increment=sg_global_step.ph_increment)
        self.op_initialize = self.Op(sg_initialize)


# Policy run by Agent(s)
class AgentModel(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_network = Network()

        sg_loss = loss.DA3CLoss(sg_network.actor, sg_network.critic,
                da3c_config.config.entropy_beta)
        sg_gradients = layer.Gradients(sg_network.weights, loss=sg_loss,
                                       norm=da3c_config.config.gradients_norm_clipping)

        # Expose public API
        self.op_assign_weights = self.Op(sg_network.weights.assign,
                weights=sg_network.weights.ph_weights)
        if da3c_config.config.use_lstm:
            self.lstm_zero_state = sg_network.lstm_zero_state
            self.op_get_action_value_and_lstm_state = self.Ops(sg_network.actor, sg_network.critic,
                    sg_network.lstm_state, state=sg_network.ph_state,
                    lstm_state=sg_network.ph_lstm_state, lstm_step=sg_network.ph_lstm_step)
            self.op_compute_gradients = self.Op(sg_gradients.calculate,
                    state=sg_network.ph_state, action=sg_loss.ph_action,
                    value=sg_loss.ph_value, discounted_reward=sg_loss.ph_discounted_reward,
                    lstm_state=sg_network.ph_lstm_state, lstm_step=sg_network.ph_lstm_step)
        else:
            self.op_get_action_and_value = self.Ops(sg_network.actor, sg_network.critic,
                    state=sg_network.ph_state)
            self.op_compute_gradients = self.Op(sg_gradients.calculate,
                    state=sg_network.ph_state, action=sg_loss.ph_action,
                    value=sg_loss.ph_value, discounted_reward=sg_loss.ph_discounted_reward)


if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, AgentModel)
