from relaax.common.python.config.loaded_config import options

config = options.get('algorithm')

for key, value in [('use_convolutions', [])]:
    if not hasattr(config, key):
        setattr(config, key, value)

config.output.scale = options.get('algorithm/output/scale', 1.0)
config.optimizer = options.get('algorithm/optimizer', 'Adam')
config.use_icm = options.get('algorithm/use_icm', False)
config.use_gae = options.get('algorithm/use_gae', False)
config.output.action_high = options.get('algorithm/output/action_high', 1.0)
config.output.action_low = options.get('algorithm/output/action_low', -1.0)
config.gradients_norm_clipping = options.get('algorithm/gradients_norm_clipping', 0.0)
