import tensorflow as tf

import relaax.algorithm_base.parameter_server_base

from . import network


class ParameterServer(relaax.algorithm_base.parameter_server_base.ParameterServerBase):
    # TODO: implement all abstract methods
    def __init__(self, config, saver, metrics):
        policy_net, value_net = network.make(config)
        obs_filter, reward_filter = network.make_filters(config)

        initialize = tf.variables_initializer(tf.global_variables())
        self._session = tf.Session()

        policy, baseline = network.make_head(config, policy_net, value_net, self._session)
        trpo_updater = network.make_trpo(policy, config, self._session)

        self._session.run(initialize)
