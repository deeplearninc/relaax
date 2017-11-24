from relaax.common.python.config.loaded_config import options

config = options.get('algorithm')


for key, value in [('use_convolutions', []),
                   ('activation', 'tanh'),
                   ('return_prob', False)]:
    if not hasattr(config, key):
        setattr(config, key, value)
