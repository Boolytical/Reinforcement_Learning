import argparse

from plotter import LearningCurvePlot
from dqn import act_in_env


# Test the dqn agent with given parameter values
def test_dqn_agent():
    # The episode terminates if (pole angle greater than -12/12) or (cart position greater than -2.4,2.4) or (episode length exceeds 500)
    # Goal: Keep up the pole for 500 timesteps (as long as possible), if done=True too soon, then reward should be negative?
    n_episodes = 200   # number of episodes the agent will go through
    n_timesteps = 500   # number of timesteps one episode can maximally contain

    policies = ['egreedy', 'softmax']

    param_dict = {
        'alpha': 0.001, # learning rate
        'gamma': 0.99,  # discount factor
        'policy': policies[0],    # exploration strategy
        'epsilon': 1.0, # initially, the agent will always choose a random action (exploration)
        'epsilon_min': 0.001,   # minimum value of exploration
        'epsilon_decay_rate': 0.001,    # exploration behavior is gradually replaced by exploitation behavior
        'tau': 0.5, # for softmax exploration strategy
        'max_replays': 2000,    # only a given amount of memory instants can be saved
        'batch_size': 32,    # number of samples from the memory that is used to fit the dnn model
        'target_network': True  # has a target network (True) or not (False)
    }

    return act_in_env(n_episodes=n_episodes, n_timesteps=n_timesteps, param_dict=param_dict)


# Determine which DQN agent is used with experiment
def determine_experiment(all_variations: bool, experience_replay: bool, target_network: bool):

    if all_variations:
        print('Run experiment on all DQN agent variations')
    
    # For experiments below, all parameter experiments are performed
    elif experience_replay and not target_network:
        print('Run experiment on DQN-ER agent')

    elif target_network and not experience_replay:
        print('Run experiment on DQN-TN agent')

    elif experience_replay and target_network:
        print('Run experiment on DQN-ER-TN agent')

        SingleRunPlot = LearningCurvePlot(title='DQN-ER-TN')

        # TODO build further on this by passing different parameter values to this function
        all_rewards_of_run = test_dqn_agent()
        SingleRunPlot.add_curve(all_rewards_of_run)
        SingleRunPlot.save(filename='dqn_er_tn.png')

    else:
        print('Run experiment on DQN agent')


if __name__ == '__main__':
    # https://docs.python.org/3/library/argparse.html#module-argparse
    parser = argparse.ArgumentParser(prog='RLA2', description='DQN Experiments')
    
    # By default, the DQN does not have experience replay and target network
    parser.add_argument('--all_variations', action='store_true', dest='all_variations', help='Comparison between all DQN variations.')
    parser.add_argument('--experience_replay', action='store_true', dest='experience_replay', help='Add to the DQN agent experience replay (This argument will be ignored when --all_experiments is parsed).')
    parser.add_argument('--target_network', action='store_true', dest='target_network', help='Add to the DQN agent an additional target network (This argument will be ignored when --all_experiments is parsed).')

    args = parser.parse_args()

    determine_experiment(args.all_variations, args.experience_replay, args.target_network)
