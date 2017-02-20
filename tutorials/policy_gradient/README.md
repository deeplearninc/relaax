### RELAAX tutorial based on simple Policy Gradient

Any RELAAX algorithm should be divided into 4 parts:
 - Client: some simulated environment to intersct;
 - Agent: worker connected end-to-end to the Client;
 - Parameter Server: aggregates results from all Agents;
 - Bridge: transport specifications between Agent and Parameter Server.

We'll focused on the last three points to implement some simple Policy Gradient algorithm.
