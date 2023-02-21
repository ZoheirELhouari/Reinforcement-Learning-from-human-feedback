# P1_El_Houari

# Reinforcement learning from human feedback


## Environment.py
the file contain the code of for all the functionalties of the environment from reset(), step(), render(). 
the enviroment requires the library pygame

## Network.py 
the file contain the two following neural networks: 
* **PolicyGradientNetwork**: the network is used to maintain a policy that interacts with the environment to produce a set of actions that has a higher sum of rewards . 
* **Rewarder**:  the network is used to estimate the reward function of a given state, the parameters are updated by fitting the human preferences as mentioned in the paper: https://proceedings.neurips.cc/paper/2017/file/d5e2c0adad503c91f91df240d0cd4e49-Paper.pdf

## Agent.py
the file contain an implimentation of the reinforce algorithm which it is used here to update policy parameters with the gradiant of the expected utility in an episode.

## Main.py
The file contain a main class where i run the experiment, the policy goes and create two sets of trajectories which are presented for the human comparison, the parameters of the reward function estimate are optimized to minimize the cross-entropy loss between the agent predictions and the actual human


## Requirements 
    pip install tensorflow== 2.9.1
    pip install pygame
    pip install tensorflow-probability== 0.17.0
    pip install numpy 
    pip install keras==2.9.0
## Run 
    python main.py
