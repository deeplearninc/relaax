from relaax.common.python.config.loaded_config import options
from argparse import Namespace
import random


config = options.get('algorithm')
config.seed = options.get('algorithm/seed', random.randrange(1000000))

config.input.universe = options.get('algorithm/input/universe', True)
config.activation = options.get('algorithm/activation', 'tanh')
config.lam = options.get('algorithm/lambda', 1.00)
config.entropy = options.get('algorithm/entropy', None)
config.l2_coeff = options.get('algorithm/l2_coeff', None)
config.critic_scale = options.get('algorithm/critic_scale', 0.25)

config.combine_gradient = options.get('algorithm/combine_gradient', 'fifo')
config.gradients_norm_clipping = options.get('algorithm/gradients_norm_clipping', False)

config.avg_in_num_batches = options.get('algorithm/avg_in_num_batches', 10)
config.use_filter = options.get('algorithm/use_filter', False)
config.use_lstm = options.get('algorithm/use_lstm', False)
config.use_icm = options.get('algorithm/use_icm', False)
config.norm_adv = options.get('algorithm/normalize_advantage', False)
config.vf_clipped_loss = options.get('algorithm/vf_clipped_loss', False)

config.policy_iterations = options.get('algorithm/policy_iterations', 5)
config.value_func_iterations = options.get('algorithm/value_func_iterations', 5)
config.mini_batch = options.get('algorithm/mini_batch', None)

# ICM parameters
if not hasattr(config, 'icm'):
    config.icm = options.get('algorithm/icm', Namespace())
config.icm.nu = options.get('algorithm/icm/nu', 0.8)
config.icm.beta = options.get('algorithm/icm/beta', 0.2)
config.icm.lr = options.get('algorithm/icm/lr', 1e-3)

# Adam parameters
if not hasattr(config, 'optimizer'):
    config.optimizer = options.get('algorithm/optimizer', Namespace())
config.optimizer.epsilon = options.get('algorithm/optimizer/epsilon', 1e-5)

config.schedule = options.get('algorithm/schedule', 'linear')
config.schedule_step = options.get('algorithm/schedule_step', 'update')     # update | environment
config.max_global_step = options.get('algorithm/max_global_step', 1e7)
