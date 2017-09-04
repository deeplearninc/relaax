from relaax.common.python.config.loaded_config import options

config = options.get('algorithm')

for key, value in [('use_convolutions', [])]:
    if not hasattr(config, key):
        setattr(config, key, value)

config.optimizer = options.get('algorithm/optimizer', 'Adam')
config.gradients_norm_clipping = options.get('algorithm/gradients_norm_clipping', False)

# check hidden sizes
if len(config.hidden_sizes) > 0:
    if config.hidden_sizes[-1] % 2 != 0:
        raise ValueError("Number of outputs in the last hidden layer must be divisible by 2")
