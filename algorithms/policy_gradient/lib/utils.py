import logging
import tensorflow as tf

from relaax.common.python.config.loaded_config import options

log = logging.getLogger(__name__)


def assemble_and_show_graphs(agent, parameter_server):
    with tf.variable_scope('parameter_server'):
        parameter_server()
    with tf.variable_scope('agent'):
        agent()
    log_dir = options.get("agent/log_dir", "log/")
    log.info(('Writing TF summary to %s. '
              'Please use tensorboad to watch.') % log_dir)
    tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())
