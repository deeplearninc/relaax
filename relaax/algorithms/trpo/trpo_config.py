from argparse import Namespace

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

if not hasattr(config, 'PPO'):
    config.PPO = options.get('algorithm/PPO', Namespace())
config.PPO.clip_e = options.get('algorithm/PPO/clip_e', 0.2)
config.PPO.learning_rate = options.get('algorithm/PPO/learning_rate', 0.001)
config.PPO.n_epochs = options.get('algorithm/PPO/n_epochs', 4)
config.PPO.minibatch_size = options.get('algorithm/PPO/minibatch_size', None)

if not hasattr(config, 'TRPO'):
    config.TRPO = options.get('algorithm/TRPO', Namespace())
config.TRPO.cg_damping = options.get('algorithm/TRPO/cg_damping', 0.1)
config.TRPO.max_kl = options.get('algorithm/TRPO/max_kl', 0.01)
config.TRPO.linesearch_type = options.get('algorithm/TRPO/linesearch_type', 'Origin')  # 'Origin' | 'Adaptive'
