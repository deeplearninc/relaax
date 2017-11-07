from relaax.common.python.config.loaded_config import options
from argparse import Namespace


config = options.get('algorithm')

config.input.universe = options.get('algorithm/input/universe', True)

config.use_lstm = options.get('algorithm/use_lstm', False)
config.use_icm = options.get('algorithm/use_icm', False)

config.policy_iterations = options.get('algorithm/policy_iterations', 5)
config.value_func_iterations = options.get('algorithm/value_func_iterations', 5)

# ICM parameters
if not hasattr(config, 'icm'):
    config.icm = options.get('algorithm/icm', Namespace())
config.icm.nu = options.get('algorithm/icm/nu', 0.8)
config.icm.beta = options.get('algorithm/icm/beta', 0.2)
config.icm.lr = options.get('algorithm/icm/lr', 1e-3)
