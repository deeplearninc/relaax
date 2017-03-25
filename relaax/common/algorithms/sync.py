from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# update PS with leaned policy
def update_shared_parameters(obj, partial_gradients):
    obj.ps.run(
        "apply_gradients", feed_dict={"gradients": partial_gradients})


# reload policy weights from PS
def load_shared_parameters(obj):
    weights = obj.ps.run("weights")
    obj.sess.run(
        obj.nn.assign_weights,
        feed_dict={obj.nn.shared_wights: weights})
