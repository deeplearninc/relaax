import tensorflow as tf

from relaax.common.algorithms import subgraph
from relaax.common.algorithms.lib import graph
from relaax.common.algorithms.lib import utils


class Gradients(subgraph.Subgraph):
    def build_graph(self, weights, loss=None, optimizer=None, norm=False, batch_size=None, grad_ys=None):
        if loss is not None:
            gradients = tf.gradients(loss.node, list(utils.Utils.flatten(weights.node)), grad_ys)
            gradients = [tf.check_numerics(g, 'gradient_%d' % i) for i, g in enumerate(gradients)]
            if batch_size is not None:
                gradients = [g / float(batch_size) for g in gradients]

            # store gradients global norm before clipping
            self.global_norm = tf.global_norm(gradients)

            # clip gradients after global norm has been stored
            if norm:
                gradients, _ = tf.clip_by_global_norm(gradients, norm)
            self.calculate = graph.TfNode(utils.Utils.reconstruct(gradients, weights.node))
        if optimizer is not None:
            self.ph_gradients = graph.Placeholders(weights)
            self.apply = graph.TfNode(optimizer.node.apply_gradients(
                    utils.Utils.izip(self.ph_gradients.checked, weights.node)))


class AdamOptimizer(subgraph.Subgraph):
    def build_graph(self, learning_rate=0.001):
        return tf.train.AdamOptimizer(learning_rate=learning_rate)


class RMSPropOptimizer(subgraph.Subgraph):
    def build_graph(self, learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10):
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate.node, decay=decay, momentum=momentum,
                                         epsilon=epsilon)
