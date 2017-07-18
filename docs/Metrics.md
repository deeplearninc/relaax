## [Metrics](../README.md#contents)
> click on title to go to contents
- [Metrics API](#metrics_api)
- [Available metrics](#available_metrics)

### [Metrics API](#metrics_api)
Each component of RELAAX has access to metrics API. relaax.client.rlx_client.RlxClient exposes metrics property for environment, Agent and ParameterServer have metrics properties for both parts of any algorithm. In all cases metrics implements the same API described by relaax.server.common.metrics.Metrics:

```python
metrics.scalar('metric_name', scalar_value, x=optional_argument)
metrics.histogram('metric_name', tensor_value, x=optional_argument)
```
Each call to any method stores additional (argument, value) pair under given metric name. Each metrics stores functional mapping. You can view and analyze this mappings in TensorBoard or server output. If you omit x parameter RELAAX will use current global_step of algorithm instead.
`metrics.scalar` is to store scalar mappings (performance counters, rewards and so on).
`metrics.histogram` is to store tensor metrics. TensorBoard will show such metrics as histogram.

### [Available metrics](#available_metrics)

  episode_reward
  server_latency
  client_latency
  action (mu, sigma2) weights, gradients?
  critic
  log_prob (gaussian_nll)
  entropy
