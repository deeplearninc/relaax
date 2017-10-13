## [Metrics](../README.md#contents)
> click on title to go to contents
- [Metrics API](#metrics_api)
- [Available metrics](#available_metrics)
- [How to use metrics](#how_to_use_metrics)

<a href="#metrics_api"></a>
### Metrics API
Each component of RELAAX has access to metrics API. relaax.client.rlx_client.RlxClient exposes metrics property for environment, Agent and ParameterServer have metrics properties for both parts of any algorithm. In all cases metrics implements the same API described by relaax.server.common.metrics.Metrics:

```python
metrics.scalar('metric_name', scalar_value, x=optional_argument)
metrics.histogram('metric_name', tensor_value, x=optional_argument)
```
Each call to any method stores (argument, value) pair under given metric name. Each metric is functional mapping from agruments to values. You can view and analyze this mappings in TensorBoard or server output. If you omit argument (x) parameter RELAAX will use current global_step of algorithm instead.
`metrics.scalar` is to store scalar mappings (performance counters, rewards and so on).
`metrics.histogram` is to store tensor metrics. TensorBoard will show such metrics as histogram.

<a href="#available_metrics"></a>
### Available metrics

- episode_reward
- server_latency
- client_latency
- action (mu, sigma2) weights, gradients?
- critic
- log_prob (gaussian_nll)
- entropy

<a href="#how_to_use_metrics"></a>
### How to use metrics

When you first use metrics on your application open app.yaml file. In this file enable all available metrics, define metrics output directory and enable logging to console output:
```yaml
relaax-metrics-server:
  enable_unknown_metrics: true  # enable metrics if it is not mentioned in list below
  metrics:                      # enable standard (predefined) metrics
    episode_reward: true
    server_latency: true
    action: true
    mu: true
    sigma2: true
    critic: true

  metrics_dir: logs/metrics     # where to store metrics (directory for tensorboard)
  log_metrics_to_console: true  # enable logging metrics to console output
```

Now run your application and check that you can see metrics in server output. If you can see then run tensorboard in separate console with `tensorboard --logdir logs/metrics`. Check tensorboard output for URL to open in browser. Open it there and navigate to scalar section. In one minute or so you can see graphical representation of your metrics. It is updated every 30s.

If everything is OK in Tensorboard you can disable logging metrics to console. This is to speed up learning and avoid console clutter with a lot of useless data: 
```yaml
relaax-metrics-server:
  log_metrics_to_console: false  # enable logging metrics to console output
```

Number of enabled metrics affects learning performace. More metrics slows down learning and vice versa. So keep metrics enables only if you need them.

If you are going to restart learning please consider removal of logs/metrics directory. Otherwise old and new metrics will be mixed in the same graph rendering general picture unreadable.
