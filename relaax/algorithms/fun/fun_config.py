from relaax.common.python.config.loaded_config import options

config = options.get('algorithm')

for key, value in [('use_convolutions', [])]:
    if not hasattr(config, key):
        setattr(config, key, value)
