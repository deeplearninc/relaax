import argparse
# from algos.misc_utils import update_argument_parser, GENERAL_OPTIONS
from algos import *     # update_argument_parser(misc_utils), GENERAL_OPTIONS(misc_utils)
import sys
import os
import shutil
# from algos import core
import gym
import logging
from tabulate import tabulate
from collections import defaultdict  # can be replaced

import tensorflow as tf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    update_argument_parser(parser, GENERAL_OPTIONS)
    parser.add_argument("--env", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--plot", action="store_true")
    args, _ = parser.parse_known_args([arg for arg in sys.argv[1:] if arg not in ('-h', '--help')])

    env = gym.make(args.env)
    env_spec = env.spec
    mondir = args.outfile + ".dir"
    if os.path.exists(mondir):
        shutil.rmtree(mondir)
    os.mkdir(mondir)
    env.monitor.start(mondir, video_callable=None if args.video else VIDEO_NEVER)

    agent_ctor = get_agent_cls(args.agent)
    update_argument_parser(parser, agent_ctor.options)
    args = parser.parse_args()
    if args.timestep_limit == 0:
        args.timestep_limit = env_spec.timestep_limit
    cfg = args.__dict__
    np.random.seed(args.seed)

    session = tf.Session()
    agent = agent_ctor(env.observation_space, env.action_space, cfg, session)   # SESSION !!! --> need ADD
    if args.use_hdf:
        hdf, diagnostics = prepare_h5_file(args)
    gym.logger.setLevel(logging.WARN)

    COUNTER = agent.restore()
    diagnostics = defaultdict(list)
    # saver = tf.train.Saver()

    def callback(stats):
        global COUNTER
        COUNTER += 1
        # Print stats
        print("*********** Iteration %i ****************" % COUNTER)
        print(tabulate(filter(lambda (k, v): np.asarray(v).size == 1, stats.items())))
        # Store to hdf5
        if args.use_hdf:
            for (stat, val) in stats.items():
                if np.asarray(val).ndim == 0:
                    diagnostics[stat].append(val)
                else:
                    assert val.ndim == 1
                    diagnostics[stat].extend(val)
            """if args.snapshot_every and ((COUNTER % args.snapshot_every == 0) or (COUNTER == args.n_iter)):
                hdf['/agent_snapshots/%0.4i' % COUNTER] = np.array(cPickle.dumps(agent, -1))"""
        # Plot
        if args.plot:
            print('Start animation...')
            animate_rollout(env, agent, min(500, args.timestep_limit))
        if COUNTER % 100 == 0:
            agent.save(COUNTER)
            # save_model(session, saver, COUNTER)

    run_policy_gradient_algorithm(env, agent, callback=callback, usercfg=cfg)

    env.monitor.close()
    # save_model(session, saver, args.n_iter)
    agent.save(COUNTER)
