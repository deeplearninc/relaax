# Introduction

We use DeepMind Lab build/run script (using bazel) to start Lab instance. To avoid changing Lab build scripts we replace random-agent target .py file with our entry point.  

In order to run multiple instances of the DeepMind Lab we run them inside docker containers (see start-container.py). This allow to execute number of bazel instances in parallel. (Dockerfile mostly based on one developed by @frankcarey - https://github.com/deepmind/lab/pull/24)

You may decide to run Lab directly on the Host. For that you may replace start-container on start-lab in the app.yaml. This will start single instance of bazel.  

In order to connect to RLX server from the docker container, we have to know IP address of the Host. Internally, we are using following script to get that IP:
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


# Use custom maps

By default, we run Lab with maps set up for `random_agent` build target. You may overwrite these maps with your own. See `environment/custom-map` folder as an example.

To use custom map, set `level_script` in `environment` section of the `app.yaml` to use custom level maps:
```yaml
environment:
  level_script: environment/custom-map/custom_map
```

## Example map

See `custom-map/t_maze`. This map has three pickups:
- `P`: respawn location
- `A`: -1 reward == `negative_one`
- `G`: +1 reward == `positive_one`
    
There are `2 custom` pickups: `negative_one` & `positive_one`
    
### Pickups script

To add some `custom` pickups we have to define them in our `pickups.lua` file.

Custom pickups may look like as these:
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

### make_map script

This scripts creates your own map from `txt` description and locates your pickups.

```lua
local pickups = {
    A = 'negative_one',
    G = 'positive_one',
    P = 'info_player_start'
}
```
