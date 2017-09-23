from relaax.common.python.config.loaded_config import options
from argparse import Namespace

config = options.get('algorithm')

for key, value in [('use_convolutions', [])]:
    if not hasattr(config, key):
        setattr(config, key, value)

config.output.scale = options.get('algorithm/output/scale', 1.0)
config.critic_scale = options.get('algorithm/critic_scale', 1.0)

config.output.loss_type = options.get('algorithm/output/loss_type', 'Normal')
config.optimizer = options.get('algorithm/optimizer', 'Adam')

config.hogwild = options.get('algorithm/hogwild', False)
config.use_icm = options.get('algorithm/use_icm', False)

config.use_gae = options.get('algorithm/use_gae', False)
config.gae_lambda = options.get('algorithm/gae_lambda', 1.00)

config.output.action_high = options.get('algorithm/output/action_high', [])
config.output.action_low = options.get('algorithm/output/action_low', [])

config.gradients_norm_clipping = options.get('algorithm/gradients_norm_clipping', False)
config.input.universe = options.get('algorithm/input/universe', True)

# ICM parameters
if not hasattr(config, 'icm'):
    config.icm = options.get('algorithm/icm', Namespace())
config.icm.nu = options.get('algorithm/icm/nu', 0.8)
config.icm.beta = options.get('algorithm/icm/beta', 0.2)
config.icm.lr = options.get('algorithm/icm/lr', 1e-3)
