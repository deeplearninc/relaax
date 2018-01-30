from relaax.common.python.config.loaded_config import options
from argparse import Namespace
import random

config = options.get('algorithm')

config.seed = options.get('algorithm/seed', random.randrange(1000000))
config.avg_in_num_batches = options.get('algorithm/avg_in_num_batches', 10)

for key, value in [('use_convolutions', [])]:
    if not hasattr(config, key):
        setattr(config, key, value)

config.input.history = options.get('algorithm/input/history', 1)
config.input.universe = options.get('algorithm/input/universe', True)

config.activation = options.get('algorithm/activation', 'relu')
config.lstm_type = options.get('algorithm/lstm_type', 'Basic')  # Basic | Dilated
config.lstm_num_cores = options.get('algorithm/lstm_num_cores', 8)

config.max_global_step = options.get('algorithm/max_global_step', 5e7)
config.use_linear_schedule = options.get('algorithm/use_linear_schedule', False)

config.initial_learning_rate = options.get('algorithm/initial_learning_rate', 1e-4)
config.learning_rate_end = options.get('algorithm/learning_rate_end', 0.0)

config.optimizer = options.get('algorithm/optimizer', 'Adam')   # Adam | RMSProp
# RMSProp default parameters
if not hasattr(config, 'RMSProp'):
    config.RMSProp = options.get('algorithm/RMSProp', Namespace())
config.RMSProp.decay = options.get('algorithm/RMSProp/decay', 0.99)
config.RMSProp.epsilon = options.get('algorithm/RMSProp/epsilon', 0.1)

config.policy_clip = options.get('algorithm/policy_clip', False)
config.critic_clip = options.get('algorithm/critic_clip', False)

config.norm_adv = options.get('algorithm/normalize_advantage', False)

config.output.loss_type = options.get('algorithm/output/loss_type', 'Normal')   # Normal | Expanded | Extended
config.gradients_norm_clipping = options.get('algorithm/gradients_norm_clipping', False)

config.entropy_beta = options.get('algorithm/entropy_beta', 0.01)
config.entropy_type = options.get('algorithm/entropy_type', 'Gauss')  # Gauss | Origin

config.gae_lambda = options.get('algorithm/gae_lambda', 1.00)
config.use_filter = options.get('algorithm/use_filter', False)

config.hogwild = options.get('algorithm/hogwild', False)
config.use_icm = options.get('algorithm/use_icm', False)

config.output.scale = options.get('algorithm/output/scale', 1.0)
config.critic_scale = options.get('algorithm/critic_scale', 1.0)

config.output.action_high = options.get('algorithm/output/action_high', [])
config.output.action_low = options.get('algorithm/output/action_low', [])

config.combine_gradients = options.get('algorithm/combine_gradients', 'fifo')
config.num_gradients = options.get('algorithm/num_gradients', 4)
config.dc_lambda = options.get('algorithm/dc_lambda', 0.05)
config.dc_history = options.get('algorithm/dc_history', 20)

# ICM default parameters
if not hasattr(config, 'icm'):
    config.icm = options.get('algorithm/icm', Namespace())
config.icm.nu = options.get('algorithm/icm/nu', 0.8)
config.icm.beta = options.get('algorithm/icm/beta', 0.2)
config.icm.lr = options.get('algorithm/icm/lr', 1e-3)

# KAF default parameters
if not hasattr(config, 'KAF'):
    config.KAF = options.get('algorithm/KAF', Namespace())
config.KAF.boundary = options.get('algorithm/KAF/boundary', 2.0)
config.KAF.size = options.get('algorithm/KAF/size', 20)
config.KAF.kernel = options.get('algorithm/KAF/kernel', 'rbf')  # rbf | rbf2d
config.KAF.gamma = options.get('algorithm/KAF/gamma', 1.0)
