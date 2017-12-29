from relaax.common.python.config.loaded_config import options

config = options.get('algorithm')

for key, value in [('use_convolutions', [])]:
    if not hasattr(config, key):
        setattr(config, key, value)

config.activation = options.get('algorithm/activation', 'tanh')
config.return_probs = options.get('algorithm/return_probs', False)

config.timesteps_per_batch = options.get('algorithm/PG_OPTIONS/timesteps_per_batch', None)
config.episodes_per_batch = options.get('algorithm/PG_OPTIONS/episodes_per_batch', 5)

config.avg_in_num_batches = options.get('algorithm/avg_in_num_batches', 2)

config.PPO.minibatch_size = options.get('algorithm/PPO/minibatch_size', None)
config.PPO.n_epochs = options.get('algorithm/PPO/n_epochs', 4)

config.TRPO.linesearch_type = options.get('algorithm/TRPO/linesearch_type', 'Origin')  # 'Origin' | 'Adaptive'
