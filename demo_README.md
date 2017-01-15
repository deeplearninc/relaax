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
docker pull 4skynet/lab-rlx
```

* Open new terminal window, navigate to training directory and run honcho:
```bash
honcho -f ../relaax/config/da3c_lab_demo.Procfile start
```

* Use `ifconfig` command to find IP of your localhost. Remember it.

* Open new terminal window, navigate to training directory and run environment inside gym docker image. Use sudo if needed.
```bash
docker run -ti -p 6080:6080 4skynet/lab-rlx
```

* Open http://127.0.0.1:6080/vnc.html URL in browser.
You will see web form to enter your credentials. Leave all fields intact and press 'Connect'.
You will see LXDE Linux desktop.
Press "Start" button -> "Accessories" -> "LXTerminal".
Run `(cd /opt/lab/bazel-bin/random_agent.runfiles/org_deepmind_lab && ./random_agent --rlx-server <LOCALHOST_IP>:7001 --display true)`.

* Browse TensorBoard output using `http://localhost:6006` URL.
