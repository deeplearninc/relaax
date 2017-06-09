### Building information

We build the `random_agent` target from bazel for DeepMind Lab by default.

Since that the building directory should be as follows:

`path_to_lab_cloned_dir/bazel-bin/random_agent.runfiles/org_deepmind_lab/`

### Run a client instance

Default DeepMind'd `random_agent` client locates here:

`org_deepmind_lab/python/random_agent.py`

To run our client we can replace this file by ours.

If you create your app next to `relaax` package, 
code can looks like as follows:
(you have to provide path to your `training.py`)
```python
#!/usr/bin/env python
import relaax
import sys
import os

path_to_lab_client = \
    str(relaax.__path__[:-6]) + 'your_app_dir/environment/training.py'
# or you can provide an absolute path to your app instead
args = [path_to_lab_client] + sys.argv[1:]
os.execv(path_to_lab_client, args)
```

You also have to set `executable` rights for `training.py`

### For Mac users

We have to run DeepMind Lab inside docker container. In order to connect to Host from the docker container, we have to know IP address of the Host. Internally, we are using following script to get that IP:
```bash
ifconfig | grep -E "([0-9]{1,3}\.){3}[0-9]{1,3}" | grep -v 127.0.0.1 | awk '{ print $2 }' | cut -f2 -d: | head -n1
``` 
You may decide to use loopback alias instead as described here:  
https://docs.docker.com/docker-for-mac/networking/#use-cases-and-workarounds

Set that alias using, for example like this:
```bash
sudo ifconfig lo0 alias 123.123.123.123/24
```
and replace RLX server binding to 123.123.123.123 (or whatever IP address you used) in app.yaml

Note: loopback alias is not persistent and should be reset after each Host reboot 

### Use a custom map

#### Location of main read_map script

Firstly, you have to set option '--level_script'
to location of you main script, which reads the map.

For example, if you script locates where:

`org_deepmind_lab/baselab/game_scripts/tests/read_map.lua`

You '--level_script' option should be: 'tests/read_map'

#### Location of map scheme

If you creates a `txt` map with name `t_maze` you can locates it there:

`org_deepmind_lab/t_maze`

This map has three pickups:
- `P`: respawn location
- `A`: -1 reward == `negative_one`
- `G`: +1 reward == `positive_one`
    
There are `2 custom` pickups related to `negative_one` & `positive_one`
    
#### Location of pickups script

To add some `custom` pickups we have to define them in our `pickups.lua` file.

You should place it there:

`org_deepmind_lab/baselab/game_scripts/common/pickups.lua`

Custom pickups can looks like as follows:
```lua
pickups.defaults = {
  negative_one = {
      name = 'Lemon',
      class_name = 'lemon_reward',
      model_name = 'models/lemon.md3',
      quantity = -1,
      type = pickups.type.kGoal
  },
  positive_one = {
      name = 'Goal',
      class_name = 'goal',
      model_name = 'models/goal_object_02.md3',
      quantity = 1,
      type = pickups.type.kGoal
  }
}
```

#### Location of make_map script

This scripts creates your own map from `txt` description and locates your pickups.

```lua
local pickups = {
    A = 'negative_one',
    G = 'positive_one',
    P = 'info_player_start'
}
```

You should place it there:

`org_deepmind_lab/baselab/game_scripts/common/make_map.lua`

### Finally

To run DeepMind Lab client from `random_agent` target you have to
run `random_agent` file from `org_deepmind_lab` root which
intercepts `org_deepmind_lab/python/random_agent.py` replaced before.

```bash
cd /lab/bazel-bin/random_agent.runfiles/org_deepmind_lab
./random_agent&
```
