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

config.combine_gradients = options.get('algorithm/combine_gradients', 'fifo')
config.num_gradients = options.get('algorithm/num_gradients', 4)
config.dc_lambda = options.get('algorithm/dc_lambda', 0.05)
config.dc_history = options.get('algorithm/dc_history', 20)

config.avg_in_num_batches = options.get('algorithm/avg_in_num_batches', 10)
