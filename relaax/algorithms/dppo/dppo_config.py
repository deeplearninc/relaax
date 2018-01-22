from relaax.common.python.config.loaded_config import options
from argparse import Namespace
import random


config = options.get('algorithm')
config.seed = options.get('algorithm/seed', random.randrange(1000000))

config.input.universe = options.get('algorithm/input/universe', True)
config.input.history = options.get('algorithm/input/history', 1)

for key, value in [('use_convolutions', [])]:
    if not hasattr(config, key):
        setattr(config, key, value)

config.avg_in_num_batches = options.get('algorithm/avg_in_num_batches', 10)
config.activation = options.get('algorithm/activation', 'tanh')

config.lam = options.get('algorithm/lambda', 1.00)
config.entropy = options.get('algorithm/entropy', None)
config.l2_coeff = options.get('algorithm/l2_coeff', None)
config.critic_scale = options.get('algorithm/critic_scale', 0.25)

config.combine_gradients = options.get('algorithm/combine_gradients', 'fifo')
config.num_gradients = options.get('algorithm/num_gradients', 4)
config.dc_lambda = options.get('algorithm/dc_lambda', 0.05)
config.dc_history = options.get('algorithm/dc_history', 20)
config.gradients_norm_clipping = options.get('algorithm/gradients_norm_clipping', False)

config.use_filter = options.get('algorithm/use_filter', False)

config.use_lstm = options.get('algorithm/use_lstm', False)
config.lstm_type = options.get('algorithm/lstm_type', 'Basic')  # Basic | Dilated
config.lstm_num_cores = options.get('algorithm/lstm_num_cores', 8)

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

# KAF default parameters
if not hasattr(config, 'KAF'):
    config.KAF = options.get('algorithm/KAF', Namespace())
config.KAF.boundary = options.get('algorithm/KAF/boundary', 2.0)
config.KAF.size = options.get('algorithm/KAF/size', 20)
config.KAF.kernel = options.get('algorithm/KAF/kernel', 'rbf')  # rbf | rbf2d
config.KAF.gamma = options.get('algorithm/KAF/gamma', 1.0)
