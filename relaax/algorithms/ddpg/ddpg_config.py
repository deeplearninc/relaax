from relaax.common.python.config.loaded_config import options

config = options.get('algorithm')

config.log_lvl = options.get('relaax-metrics-server/log_level', 'INFO')  # INFO | DEBUG | VERBOSE
config.no_ps = options.get('algorithm/no_ps', False)
config.input.history = options.get('algorithm/input/history', 1)

for key, value in [('use_convolutions', [])]:
    if not hasattr(config, key):
        setattr(config, key, value)

config.buffer_size = options.get('algorithm/buffer_size', 4*10**5)
config.batch_size = options.get('algorithm/batch_size', 64)
config.loop_size = options.get('algorithm/loop_size', 1)
config.input.history = options.get('algorithm/input/history', 1)
config.hidden_sizes = options.get('algorithm/hidden_sizes', [400, 300])  # needs at least two layers

config.actor_learning_rate = options.get('algorithm/actor_learning_rate', 1e-4)
config.critic_learning_rate = options.get('algorithm/critic_learning_rate', 1e-3)
config.tau = options.get('algorithm/tau', 1e-3)
config.use_filter = options.get('algorithm/use_filter', False)

config.l2 = options.get('algorithm/l2', True)
config.l2_decay = options.get('algorithm/l2_decay', 1e-2)

config.output.scale = options.get('algorithm/output/scale', 1.)
config.ou_noise = options.get('algorithm/ou_noise', True)

config.exploration.ou_mu = options.get('algorithm/exploration/ou_mu', .0)
config.exploration.ou_theta = options.get('algorithm/exploration/ou_theta', .15)
config.exploration.ou_sigma = options.get('algorithm/exploration/ou_sigma', .2)
config.exploration.tau = options.get('algorithm/exploration/tau', 25)
config.exploration.rnd_seed = options.get('algorithm/exploration/rnd_seed', 1234)

config.combine_gradients = options.get('algorithm/combine_gradients', 'fifo')
config.num_gradients = options.get('algorithm/num_gradients', 4)
config.dc_lambda = options.get('algorithm/dc_lambda', 0.05)
config.dc_history = options.get('algorithm/dc_history', 20)

config.avg_in_num_batches = options.get('algorithm/avg_in_num_batches', 200)
