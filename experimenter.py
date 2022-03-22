import argparse
from ast import arg

from plotter import LearningCurvePlot


# 
def run_dqn_agent():
    print('Run')


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
