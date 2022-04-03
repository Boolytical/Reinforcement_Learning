# Reinforcement Learning Assignment 2: Deep Q-Learning
By Esmee Roosenmaallen, Felix Kapulla, Rosa Zwart
*** 
The provided .py files (dqn.py, experimenter.py, plotter.py, util.py) implement a deep Q-learning agent that acts in the Cartpole environment provided by Gym, OpenAI (https://gym.openai.com/envs/CartPole-v1/). The code experiments with different parameter settings and variations of the DQN agent including or excluding the usage of a target network and replay memory.
***

## Descriptions of the files
***
* util.py           This file contains additional utility functions used in our implementation.
* plotter.py        This file contains the function that creates the learning curve plots.
* dqn.py            This file contains the DQN agent class with its relevant functions as well as the function that enables the agent acting in the provided environment.   
* experimenter.py   This file contains the functions that interpret the arguments included in the run commands and perform the accessory experiments that in the end save the relevant plots in the same working directory.
***

## How to run the code
***
There are multiple options to run the code, listed below:
* $ python experimenter.py --experience_replay --target_network         This command runs four experiments for the DQN-ER-TN agent in order to determine the parameter settings that result in the best performance. Experiment 1 tests the epsilon-greedy policy with various learning rates and epsilon decay rates. Experiment 2 is used to determine the best settings for the softmax policy by testing different values for the learning rate and tau parameter. Experiment 3 explores the performance of the agent using different gamma values. These gamma values are combined with both policies (epsilon-greedy and softmax) with the settings that have been determined to be most optimal based on observations of the previous two experiments. The last experiment, Experiment 4, tests different network architectures to see how this would influence the performance of the agent. Again, we use the found optimal parameter settings from the previous experiments and apply the architecture changes to the agent using the epsilon-greedy policy as well as the softmax policy. 
* $ python experimenter.py --all_variations         This command runs the ablation study. All DQN variations are run in the environment (DQN, DQN-ER, DQN-TN, DQN-ER-TN) with the optimal parameter settings that have been found in all parameter-tuning experiments using DQN-ER-TN (see above). 
* $ python experimenter.py         This command runs only the DQN agent with the determined optimal parameter settings found from Experiment 1 to 4 discussed above. (Not used nor discussed in the report.) It is possible to either add argument --experience_replay or --target_network (or both) to this command to run the different variations of the DQN agent individually.
***
