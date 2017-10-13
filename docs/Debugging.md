## [Debugging](../README.md#contents)
> click on title to go to contents
- [Debugging](#debbuging)
- [Profiling](#profiling)

### Debugging
Best way to debug Reinforcement Learning algorithm is to analyze metrics. Metrics shows that algorithm observes states, changes NN weights, improves loss values. Please refer [metrics documentation](Metrics.md) for further investigation.

### Profiling
Profiling is the way to gather performance information from all servers and clients participating in learning process.

To gather profiling data please ensure that your code is instrumented properly:
```python
...
from relaax.common import profiling
...
profiler = profiling.get_profiler(__name__)
...
class AgentProxy(object):
...
    @profiler.wrap
    def update(self, reward=None, state=None, terminal=False):
...
```

Most of code in RELAAX is already instrumented and ready to gather profile data. You can add more insrumenation in your code or in RELAAX code.

Next step is to enable profiling. Edit app.yaml:
```yaml
relaax-metrics-server:
  profile_dir: logs/profile

relaax-parameter-server:
  profile_dir: logs/profile

relaax-rlx-server:
  profile_dir: logs/profile
```

Now run the application. Let it work for about a minute. Then stop and check logs/profile directory. There you can find several text files: one for parameter server, one for metrics server and several files for RLX servers. Please unite and convert them into one JSON file:
```bash
<path_to_relaax>/relaax/cmdl/utils/make_profile.py logs/profile/*.txt > logs/profile/profile.json
```
To analyse profile data open `chrome://tracing/` URL in Chrome browser, press Load button and select `logs/profile/profile.json` file. You can see all your servers activity on common timeline.

Profiling is time and disk consuming activity so disable profiling if do not need it any more. Just remove or comment out `profile_dir` keys for all servers in app.yaml file.
