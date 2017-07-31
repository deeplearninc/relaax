from relaax.common.python.config.loaded_config import options

config = options.get('algorithm')
config.debug = options.get('algorithm/debug', True)

config.buffer_size = options.get('algorithm/buffer_size', 4*10**5)
config.batch_size = options.get('algorithm/batch_size', 64)
config.loop_size = options.get('algorithm/loop_size', 1)    # 4
config.input.history = options.get('algorithm/input/history', 1)
config.hidden_sizes = options.get('algorithm/hidden_sizes', [400, 300])  # needs at least two layers

config.actor_learning_rate = options.get('algorithm/actor_learning_rate', 1e-4)
config.critic_learning_rate = options.get('algorithm/critic_learning_rate', 1e-3)
config.tau = options.get('algorithm/tau', 1e-3)

config.l2 = options.get('algorithm/l2', False)
config.l2_decay = options.get('algorithm/l2_decay', 1e-2)

config.ou_noise = options.get('algorithm/ou_noise', False)

config.exploration.ou_mu = options.get('algorithm/exploration/ou_mu', .0)
config.exploration.ou_theta = options.get('algorithm/exploration/ou_theta', .15)
config.exploration.ou_sigma = options.get('algorithm/exploration/ou_sigma', .2)
config.exploration.tau = options.get('algorithm/exploration/tau', 25)
config.exploration.rnd_seed = options.get('algorithm/exploration/rnd_seed', 1234)
