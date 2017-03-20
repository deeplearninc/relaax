# Sample of wiring everything together 

This is an example of configuration to "run" sample algorithms w/ sample clients

This sample main purpose to demonstrate wiring of the client and algorithms

To run this "training" set algorithm/path to relaax algorithm location 'algorithms/sample' in sample.yaml

Start parameter server:
```bash
relaax-parameter-server --config sample.yaml
```
Start RLX server:
```bash
relaax-rlx-server --config sample.yaml
```
Run client:
```bash
python client/sample_exchange.py
```

