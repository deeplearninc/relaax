## [Quick start](#quick-start)

We recommended you use an isolated Python environment to run RELAAX. Virtualenv or Anaconda are examples. If you're using the system's python environment, you may need to run `pip install` commands with `sudo`. On OSX / macOS, we recommend using [Homebrew](http://brew.sh/) to install a current python version.

* Install <a href="https://docs.docker.com/engine/installation/" target="_blank">Docker</a>

* Clone RELAAX repo.
```bash
git clone git@github.com:deeplearninc/relaax.git
```

* Install RELAAX
```bash
cd relaax
pip install -e .
```

* Build DA3C bridge.
```bash
algorithms/da3c/bridge/bridge.sh
```

* Install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a>

* Create training directory
```bash
cd ..
mkdir training
cd training
```

* Build Docker image named gym (use sudo if needed):
```bash
docker build -f ../relaax/environments/OpenAI_Gym/Dockerfile -t gym ../relaax
```

* Open new terminal window, navigate to training directory and run parameter server
```bash
relaax-parameter-server --config ../relaax/config/da3c_gym_boxing.yaml
```

* Open new terminal window, navigate to training directory and run RLX server
```bash
relaax-rlx-server --config ../relaax/config/da3c_gym_boxing.yaml --bind 0.0.0.0:7001
```

* Use `ifconfig` command to find IP of your localhost. Remember it.

* Open new terminal window, navigate to training directory and run environment inside gym docker image. Use sudo if needed.
```bash
docker run -ti gym <LOCALHOST_IP>:7001 Boxing-v0
```

* Open new terminal window, navigate to trainin directory and run Tensorboard:
```bash
tensorboard --logdir metrics_gym_boxing
```

* Tensorboard prints URL to use. Open it in browser to exemain training progress.
