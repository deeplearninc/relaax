from relaax.common.python.config.loaded_config import options

config = options.get('algorithm')

for key, value in [('use_convolutions', [])]:
    if not hasattr(config, key):
        setattr(config, key, value)

config.avg_in_num_batches = options.get('algorithm/avg_in_num_batches', 2)
