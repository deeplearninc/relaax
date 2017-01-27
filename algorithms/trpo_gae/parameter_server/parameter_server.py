import tensorflow as tf

import relaax.algorithm_base.parameter_server_base

from . import network


class ParameterServer(relaax.algorithm_base.parameter_server_base.ParameterServerBase):
    def __init__(self, config, saver, metrics):
        self.policy_net, self.value_net = network.make(config)

        initialize = tf.variables_initializer(tf.global_variables())
        self._session = tf.Session()

        self.policy, self.baseline = network.make_head(config, self.policy_net, self.value_net, self._session)
        trpo_updater = network.make_trpo(self.policy, config, self._session)

        self._session.run(initialize)
        #self._bridge = _Bridge(config, metrics, self._network, self._session)

    def close(self):
        self._session.close()

    def restore_latest_checkpoint(self):
        pass
        #return self._saver.restore_latest_checkpoint(self._session)

    def save_checkpoint(self):
        pass
        #self._saver.save_checkpoint(self._session, self.global_t())

    def checkpoint_location(self):
        pass
        #return self._saver.location()

    def global_t(self):
        pass
        #return self._session.run(self._network.global_t)

    def bridge(self):
        pass
        #return self._bridge
