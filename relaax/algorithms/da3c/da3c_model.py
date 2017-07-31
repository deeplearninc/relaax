from __future__ import absolute_import

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import layer
from relaax.common.algorithms.lib import loss
from relaax.common.algorithms.lib import utils
from .lib import da3c_graph
from . import da3c_config
from . import icm_model


class Network(subgraph.Subgraph):
    def build_graph(self):
        conv_layer = dict(type=layer.Convolution, activation=layer.Activation.Elu,
                          n_filters=32, filter_size=[3, 3], stride=[2, 2],
                          border=layer.Border.Same)
        input = layer.Input(da3c_config.config.input, descs=[dict(conv_layer)] * 4)

        sizes = da3c_config.config.hidden_sizes
        actor_layers = [input]

        flattened_input = layer.Flatten(input)
        last_size = flattened_input.node.shape.as_list()[-1]
        if len(sizes) > 0:
            last_size = sizes[-1]

        if da3c_config.config.use_lstm:
            lstm = layer.LSTM(graph.Expand(flattened_input, 0), size=last_size)
            head = graph.Reshape(lstm, [-1, last_size])
            actor_layers.append(lstm)

            self.ph_lstm_state = lstm.ph_state
            self.ph_lstm_step = lstm.ph_step
            self.lstm_zero_state = lstm.zero_state
            self.lstm_state = lstm.state
        else:
            head = layer.GenericLayers(flattened_input,
                                       [dict(type=layer.Dense, size=size,
                                             activation=layer.Activation.Relu6) for size in sizes])
            actor_layers.append(head)

        actor = layer.Actor(head, da3c_config.config.output)
        actor_layers.append(actor)

        c1 = layer.Dense(flattened_input, 100, activation=layer.Activation.Relu6)
        critic = layer.Dense(c1, 1)
        critic_layers = (c1, critic)

        self.ph_state = input.ph_state
        self.actor = actor
        self.critic = graph.Flatten(critic)
        self.actor_weights = layer.Weights(*actor_layers)
        self.critic_weights = layer.Weights(*critic_layers)


# Weights of the policy are shared across
# all agents and stored on the parameter server
class SharedParameters(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_global_step = graph.GlobalStep()
        sg_network = Network()
        sg_actor_weights = sg_network.actor_weights
        sg_critic_weights = sg_network.critic_weights

        if da3c_config.config.optimizer == 'Adam':
            sg_actor_optimizer = graph.AdamOptimizer(da3c_config.config.actor_initial_learning_rate)
            sg_critic_optimizer = graph.AdamOptimizer(da3c_config.config.critic_initial_learning_rate)
            if da3c_config.config.use_icm:
                sg_icm_optimizer = graph.AdamOptimizer(da3c_config.config.actor_initial_learning_rate)
                sg_icm_weights = icm_model.ICM().weights
                sg_icm_gradients = layer.Gradients(sg_icm_weights, optimizer=sg_icm_optimizer)
        else:
            sg_actor_learning_rate = da3c_graph.LearningRate(sg_global_step,
                                                             da3c_config.config.actor_initial_learning_rate)
            sg_critic_learning_rate = da3c_graph.LearningRate(sg_global_step,
                                                              da3c_config.config.critic_initial_learning_rate)
            sg_actor_optimizer = graph.RMSPropOptimizer(learning_rate=sg_actor_learning_rate,
                decay=da3c_config.config.RMSProp.decay, momentum=0.0,
                epsilon=da3c_config.config.RMSProp.epsilon)
            sg_critic_optimizer = graph.RMSPropOptimizer(learning_rate=sg_critic_learning_rate,
                decay=da3c_config.config.RMSProp.decay, momentum=0.0,
                epsilon=da3c_config.config.RMSProp.epsilon)
        sg_actor_gradients = layer.Gradients(sg_actor_weights, optimizer=sg_actor_optimizer)
        sg_critic_gradients = layer.Gradients(sg_critic_weights, optimizer=sg_critic_optimizer)
        sg_initialize = graph.Initialize()

        if da3c_config.config.use_icm:
            # Expose ICM public API
            self.op_icm_get_weights = self.Op(sg_icm_weights)
            self.op_icm_apply_gradients = self.Ops(sg_icm_gradients.apply,
                                                   gradients=sg_icm_gradients.ph_gradients)

        # Expose public API
        self.op_n_step = self.Op(sg_global_step.n)
        self.op_check_weights = self.Ops(sg_actor_weights.check, sg_critic_weights.check)
        self.op_get_weights = self.Ops(sg_actor_weights, sg_critic_weights)
        self.op_apply_gradients = self.Ops(sg_actor_gradients.apply, sg_critic_gradients.apply,
                                           sg_global_step.increment,
                                           actor_gradients=sg_actor_gradients.ph_gradients,
                                           critic_gradients=sg_critic_gradients.ph_gradients,
                                           increment=sg_global_step.ph_increment)
        self.op_initialize = self.Op(sg_initialize)


# Policy run by Agent(s)
class AgentModel(subgraph.Subgraph):
    def build_graph(self):
        # Build graph
        sg_network = Network()

        sg_loss = loss.DA3CLoss(sg_network.actor, sg_network.critic, da3c_config.config)
        sg_actor_gradients = layer.Gradients(sg_network.actor_weights,
                                             loss=graph.TfNode(sg_loss.policy_loss),
                                             norm=da3c_config.config.gradients_norm_clipping)
        sg_critic_gradients = layer.Gradients(sg_network.critic_weights,
                                              loss=graph.TfNode(sg_loss.value_loss),
                                              norm=da3c_config.config.gradients_norm_clipping)

        if da3c_config.config.use_icm:
            sg_icm_network = icm_model.ICM()
            sg_icm_loss = loss.ICMLoss(sg_network.actor, sg_icm_network,
                                       da3c_config.config.ICM.alpha, da3c_config.config.ICM.beta)
            sg_icm_gradients = layer.Gradients(sg_icm_network.weights, loss=sg_icm_loss)

            # Expose ICM public API
            self.op_icm_assign_weights = self.Op(sg_icm_network.weights.assign,
                                                 weights=sg_icm_network.weights.ph_weights)
            self.op_get_intrinsic_reward = self.Ops(sg_icm_network.rew_out,
                                                    state=sg_icm_network.ph_state)
            self.op_compute_icm_gradients = self.Op(sg_icm_gradients.calculate,
                                            state=sg_icm_network.ph_state, action=sg_icm_loss.ph_action,
                                            discounted_reward=sg_icm_loss.ph_discounted_reward)

        # Expose public API
        self.op_assign_weights = self.Ops(sg_network.actor_weights.assign, sg_network.critic_weights.assign,
                                         actor_weights=sg_network.actor_weights.ph_weights,
                                         critic_weights=sg_network.critic_weights.ph_weights)

        feeds = dict(state=sg_network.ph_state, action=sg_loss.ph_action,
                     discounted_reward=sg_loss.ph_discounted_reward)

        if da3c_config.config.use_gae:
            feeds.update(dict(advantage=sg_loss.ph_advantage))

        if da3c_config.config.use_lstm:
            feeds.update(dict(lstm_state=sg_network.ph_lstm_state, lstm_step=sg_network.ph_lstm_step))
            self.lstm_zero_state = sg_network.lstm_zero_state
            self.op_get_action_value_and_lstm_state = self.Ops(sg_network.actor, sg_network.critic,
                                                               sg_network.lstm_state,
                                                               state=sg_network.ph_state,
                                                               lstm_state=sg_network.ph_lstm_state,
                                                               lstm_step=sg_network.ph_lstm_step)
        else:
            self.op_get_action_and_value = self.Ops(sg_network.actor, sg_network.critic,
                                                    state=sg_network.ph_state)

        self.op_compute_gradients = self.Ops(sg_actor_gradients.calculate, sg_critic_gradients.calculate,
                                             **feeds)


if __name__ == '__main__':
    utils.assemble_and_show_graphs(SharedParameters, AgentModel)
