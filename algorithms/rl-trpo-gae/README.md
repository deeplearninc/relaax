needs:

> keras & gym $ tabulate

run:

> python main.py --gamma=0.995 --lam=0.97 --agent=algos.agents.TrpoAgent --max_kl=0.01 --cg_damping=0.1 --activation=tanh --n_iter=2000 --seed=0 --timesteps_per_batch=50000 --env=BipedalWalkerHardcore-v2 --outfile=outdir