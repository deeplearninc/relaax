### Installation

You've to have installed OpenAI `gym`

If you don't have it, just install by follows:
```bash
$ pip install gym['all']
```

Also you have to be sure that you've installed these dependencies:
```bash
apt-get install -y python-numpy cmake zlib1g-dev libjpeg-dev libboost-all-dev gcc libsdl2-dev wget unzip git
```

Then you can install specific packages for this `doom`
```bash
$ pip install gym-pull
$ pip install ppaquette-gym-doom
```

Finally, you list of python packages should like as follows:
```bash
doom-py==0.0.15
gym>=0.8.1,<=0.9.1
gym-pull>=0.1.6,<=0.1.7
mock==2.0.0
numpy==1.12.1
olefile==0.44
pbr==3.0.1
Pillow==4.1.1
ppaquette-gym-doom==0.0.6
protobuf=>3.1.0,<=3.3.0
pyglet==1.2.4
requests==2.17.3
six==1.10.0
```
You can use higher versions of the packages above,
but it's not tested.
