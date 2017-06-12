from __future__ import absolute_import

import tensorflow as tf

from relaax.common.algorithms import subgraph


class NIter(subgraph.Subgraph):
    def build_graph(self):
        sg_collect = tf.Variable(False)
        sg_turn_collect_on = tf.assign(sg_collect, True)
        sg_turn_collect_off = tf.assign(sg_collect, False)

        sg_n_iter = tf.Variable(0, dtype=tf.int64)
        sg_next_iter = tf.assign_add(sg_n_iter, 1)

        sg_where = tf.where(sg_collect, tf.cast(-1, dtype=tf.int64), sg_n_iter)

        self.op_turn_collect_on = self.Op(sg_turn_collect_on)
        self.op_turn_collect_off = self.Op(sg_turn_collect_off)
        self.op_n_iter = self.Op(sg_where)
        self.op_next_iter = self.Op(sg_next_iter)
