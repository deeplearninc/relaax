import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import utils


class Gradients(subgraph.Subgraph):
    def build_graph(self, weights, loss=None, optimizer=None, norm=False, batch_size=None, grad_ys=None):
        if loss is not None:
            grads = tf.gradients(loss.node, list(utils.Utils.flatten(weights.node)), grad_ys)
            if batch_size is not None:
                grads = [g / float(batch_size) for g in grads]
            self.global_norm = graph.TfNode(tf.global_norm(grads))
            if norm:
                grads, _ = tf.clip_by_global_norm(grads, norm)
            grads = (tf.check_numerics(g, 'gradient_%d' % i) for i, g in enumerate(grads))
            self.calculate = graph.TfNode(utils.Utils.reconstruct(grads, weights.node))
        if optimizer is not None:
            self.ph_gradients = graph.Placeholders(weights)
            self.apply = graph.TfNode(optimizer.node.apply_gradients(
                    utils.Utils.izip(self.ph_gradients.checked, weights.node)))


class AdamOptimizer(subgraph.Subgraph):
    def build_graph(self, learning_rate=0.001):
        return tf.train.AdamOptimizer(learning_rate=learning_rate)


class RMSPropOptimizer(subgraph.Subgraph):
    def build_graph(self, learning_rate, decay, momentum, epsilon):
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate.node, decay=decay, momentum=momentum,
                                         epsilon=epsilon)
