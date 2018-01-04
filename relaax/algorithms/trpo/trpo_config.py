from relaax.common.python.config.loaded_config import options

config = options.get('algorithm')

config.timesteps_per_batch = options.get('algorithm/timesteps_per_batch', None)
config.episodes_per_batch = options.get('algorithm/episodes_per_batch', 5)
config.trpo_max_gradient_norm = options.get('algorithm/TRPO/max_gradient_norm', False)

for key, value in [('use_convolutions', []),
                   ('activation', 'tanh'),
                   ('return_prob', False)]:
    if not hasattr(config, key):
        setattr(config, key, value)
