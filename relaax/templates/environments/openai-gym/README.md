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

Then you can install specific packages for `atari`
```bash
$ pip install gym[atari]
```
Windows:
```bash
$ pip install gym
$ pip install git+https://github.com/Kojoley/atari-py.git
```

### Generate new application:
```bash
$ relaax new app_name -e openai-gym -a trpo
```

### app.yaml parameters for ATARI games with images as a state:
#### name
Can be any name from this list:

'AirRaid', 'Alien', 'Amidar', 'Assault', 'Asterix',
'Asteroids', 'Atlantis', 'BankHeist', 'BattleZone', 'BeamRider',
'Berzerk', 'Bowling', 'Boxing', 'Breakout', 'Carnival',
'Centipede', 'ChopperCommand', 'CrazyClimber', 'DemonAttack', 'DoubleDunk',
'ElevatorAction', 'Enduro', 'FishingDerby', 'Freeway', 'Frostbite',
'Gopher', 'Gravitar', 'IceHockey', 'Jamesbond', 'JourneyEscape',
'Kangaroo', 'Krull', 'KungFuMaster', 'MontezumaRevenge', 'MsPacman',
'NameThisGame', 'Phoenix', 'Pitfall', 'Pong', 'Pooyan',
'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank',
'Seaquest', 'Skiing', 'Solaris', 'SpaceInvaders', 'StarGunner',
'Tennis', 'TimePilot', 'Tutankham', 'UpNDown', 'Venture',
'VideoPinball', 'WizardOfWor', 'YarsRevenge', 'Zaxxon'

For example:

```yaml
environment:
    name: AirRaid-v0
```

#### image OR shape
Should contain image dimensions and color planes. If your state is image, use 'image' key instead of 'shape'

For example for RGB image 84x84 pixels:

```yaml
environment:
  image: [84, 84, 3]    # image dimensions (width, height, color planes)

algorithm:
  input:
    image: [84, 84, 3]  # image dimensions (width, height, color planes)

  output:
    action_size: 6                # action size for the AirRaid environment

```


You can use shape key as a synonym of image key.
