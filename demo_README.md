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

* Pull DeepMind Lab demo docker
```bash
docker pull 4skynet/lab
```

* If you are going to use trained checkpoint downloaded it from here: <a href="https://s3.amazonaws.com/dl-checkpoints/lab_demo_checkpoints.tar.gz" target="_blank">lab_demo_checkpoints.tar.gz</a>. Unpack it to training directory.

* Open new terminal window, navigate to training directory and run honcho:
```bash
honcho -f ../relaax/config/da3c_lab_demo.Procfile start
```

* Open new terminal window and run environment inside docker image. Use sudo if needed.
```bash
docker run --net="host" -ti 4skynet/lab localhost
```

* Open http://127.0.0.1:6080/vnc.html URL in browser.
You will see web form to enter your credentials. Leave all fields intact and press 'Connect'.
You will see running game.

* Browse TensorBoard output using `http://localhost:6006` URL.
