from __future__ import absolute_import

import tensorflow as tf

import relaax.server.common.session as session

import relaax.algorithms.trpo.lib.trpo_graph as trpo_graph


class TestTrpoGraph(object):
    def test_niter(self):
        n_iter = trpo_graph.NIter()
        init = tf.global_variables_initializer()
        s = session.Session(n_iter)
        s.session.run(init)

        assert s.op_n_iter() == 0
        s.op_next_iter()
        assert s.op_n_iter() == 1
        s.op_turn_collect_on()
        assert s.op_n_iter() == -1
        s.op_next_iter()
        assert s.op_n_iter() == -1
        s.op_turn_collect_off()
        assert s.op_n_iter() == 2
