# Implementation of a Client for OpenAI gym
 
 Just run main.py to start training:
 > main.py
 
 Default parameters are:
 - algo = "a3c" ("dqn" also available)
 - game = "Boxing-v0"
 - agents = "8"
 
 You can also specify these parameters manually, for example:
 
 > main.py --algo dqn --game SpaceInvaders-v0 --agents 8
 (but if you choose not async algo agents count resets to one)
 
 For playing just type in terminal `d` then `Enter`
 (if game didn't start try to repeat one more time)
 
## How to Run & Dependencies

Before you start, make sure you have installed on your system:

- python 2.7
- pip
- virtualenvs (http://docs.python-guide.org/en/latest/dev/virtualenvs/#virtualenvironments-ref)

To get started run the following commands (Linux):

- cd PROJECT_DIR
- virtualenv -p /usr/bin/python2.7 ENVIRONMENT_NAME
- source ENVIRONMENT_NAME/bin/activate
- pip install -r requirements.txt

If installation of `gym` failed try to install these dependencies:

`sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig`

Then install `gym` again:

 > pip install gym

And for full installation:

 > pip install 'gym[all]'