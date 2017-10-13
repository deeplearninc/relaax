# Samples of RELAAX applications

To run sample, navigate to sample folder and run:
```bash
relaax run
```
If sample environment implemented in JavaScript and you want to rebuild environment, run:
```bash
webpack
```

## List of samples

* simple-exchange: don't perform any training. It demonstrates:
    * wiring of the Agent, Parameter Server, and Model
    * environment and algorithm exchange of the `reward`, `state`, and `actions`
    * just for demonstration, algorithm implementation located outside of the app folder (see app.yaml)

* simple-exchange-js: don't perform any training. It demonstrates implementation of environment in JS and environment/algorithm exchange of the `reward`, `state`, and `actions`

* basic-js: implements multi-armed Bandit environment in JS and trains it with Policy Gradient
