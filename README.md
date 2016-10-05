# DeepLearn Services
Our architecture is divided by:

* [Server](/server) which manages models and its algorithms]
* [Clients](/clients) which feeds server by its data

We have 2 types of clients at this moment:

- [ALE Client](/clients/rl-client-ale)
which can emulates Atari Games

- [OpenAI's gym Client](/clients/rl-client-gym)
emulates all gym's Environments such as:
    * [Classic Control](https://gym.openai.com/envs#classic_control)
    * [Atari Games](https://gym.openai.com/envs#atari)
    * [Walkers, Landers & Racing](https://gym.openai.com/envs##box2d)
    * ant others, see the full list [there](https://gym.openai.com/envs)
    
To see how it works you should clone this repo and run separately the 
server and one of clients or both of them. You should follows the 
instructions, which you can find in appropriate [readme file](#links-to-readme-files)

### Links to Readme files
[Server's readme](/server/README.md)

[ALE Client's readme](/clients/rl-client-ale/README.md)

[GYM Client's readme](/clients/rl-client-gym/README.md)
