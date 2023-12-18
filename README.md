# SuperMarioBrosQL
CS 4701 Final Project by Eric Guo and Ben Wu

SuperMarioBrosQL is an exploratory research project into a use case for the Double 
Deep Q-Network (DDQN) algorithm on Super Mario Bros. World 1-1. We used [gym-super-mario-bros](https://pypi.org/project/gym-super-mario-bros/) 
for our display of the game, and we implemented the DDQN agent with guidance from [PyTorch](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html), [Andrew Grebenisan](https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/), and the [DDQN paper](https://arxiv.org/abs/1509.06461) itself. 

You can view our full report with background, methodology, and results here: [CS4701 Final Write-Up.pdf](https://github.com/ericguo31/SuperMarioBrosQL/files/13707708/CS4701.Final.Write-Up.pdf)

# Flowchart
<img width="845" alt="SuperMariosBrosQL - Flowchart" src="https://github.com/ericguo31/SuperMarioBrosQL/assets/71906595/9c60fb57-8fd9-4236-90bc-0d43c7126d57">

# Setup
First, install necessary requirements.

```
pip install -r requirements.txt
```

We provide functionality for human, random, and agent play.
- Human allows you, the user, to interact with the Super Mario Bros. world. Currently, only RandomStages from gym-super-mario-bros is supported. Refer to gym-super-mario-bros documentation for specific input.
- Random displays random movements for Mario. Refer to gym-super-mario-bros documentation for specific input. 
- Agent runs a pretrained or newly trained agent following the DQN/DDQN/randomized algorithm as specified. This randomized algorithm differs from random play because it is specific to our research project's objectives and has the ability to save runs.

```
python3 main.py -tm <whether you want to train the agent> -pre <whether agent is pretrained> -eps <no. of episodes to train/run agent> -alg <'ddqn', 'dqn', 'random'> -e <the environment ID to play> -m <`human` or `random`> -a <actionspace: 'nes', 'right', 'simple', 'complex'> -s <no. of steps> -S <specific stages to play in Super Mario Bros.>
```

# Citations
```
@misc{gym-super-mario-bros,
  author = {Christian Kauten},
  howpublished = {GitHub},
  title = {{S}uper {M}ario {B}ros for {O}pen{AI} {G}ym},
  URL = {https://github.com/Kautenja/gym-super-mario-bros},
  year = {2018},
}
```

