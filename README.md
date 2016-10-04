# DeepLearn Services
Our architecture is divided by:

* [Server, which manages models and its algorithms](/server)
* [Clients, which feeds server by its data](/clients)

We have 2 types of clients at this moment:

- [ALE Client](/clients/rl-client-ale)
which can emulates Atari Games

- [OpenAI's gym Client](/clients/rl-client-gym)
emulates all gym's Environments such as:
    * [Classic Control](https://gym.openai.com/envs#classic_control)
    * [Atari Games](https://gym.openai.com/envs#atari)
    * [Walkers, Landar & Racing](https://gym.openai.com/envs##box2d)
    * ant others, see the full list [there](https://gym.openai.com/envs)

### Links to Readme files
[Server's readme](/server/README.md)
[ALE Client's readme](/clients/rl-client-ale/README.md)
[GYM Client's readme](/clients/rl-client-gym/README.md)
