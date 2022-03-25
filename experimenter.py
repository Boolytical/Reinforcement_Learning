import argparse

from plotter import LearningCurvePlot
from dqn import act_in_env
import numpy as np
from plotter import smooth


# Test the dqn agent with given parameter values
def test_dqn_agent(target_network: str, n_timesteps, n_episodes, tau, learning_rate, policy):
    # The episode terminates if (pole angle greater than -12/12) or (cart position greater than -2.4,2.4) or (episode length exceeds 500)
    # Goal: Keep up the pole for 500 timesteps (as long as possible), if done=True too soon, then reward should be negative?


    param_dict = {
        'alpha': learning_rate,  # learning rate
        'gamma': 0.99,  # discount factor
        'policy': policy,  # exploration strategy
        'epsilon': 1.0,  # initially, the agent will always choose a random action (exploration)
        'epsilon_min': 0.001,  # minimum value of exploration
        'epsilon_decay_rate': 0.001,  # exploration behavior is gradually replaced by exploitation behavior
        'tau': tau,  # for softmax exploration strategy
        'max_replays': 2000,  # only a given amount of memory instants can be saved
        'batch_size': 64,  # number of samples from the memory that is used to fit the dnn model
        'target_network': target_network,  # has a target network (True) or not (False)
    }

    return act_in_env(n_episodes=n_episodes, n_timesteps=n_timesteps, param_dict=param_dict)


# Determine which DQN agent is used with experiment
def determine_experiment(all_variations: bool, experience_replay: bool, target_network: bool):

    n_episodes = 200  # number of episodes the agent will go through
    n_timesteps = 500  # number of timesteps one episode can maximally contain
    n_repititions = 3
    smoothing_window = 51

    target_network = True # if False then it is sample_wise

    policies = ['egreedy', 'softmax']
    taus = [0.1, 0.5, 1.0]
    learning_rates = [0.01, 0.1, 0.2]

    if all_variations:
        print('Run experiment on all DQN agent variations')

    # For experiments below, all parameter experiments are performed
    elif experience_replay and not target_network:
        print('Run experiment on DQN-ER agent')

    elif target_network and not experience_replay:
        print('Run experiment on DQN-TN agent')

    elif experience_replay and target_network:
        print('Run experiment on DQN-ER-TN agent')

        MultipleRunPlot = LearningCurvePlot(title=f'DQN-ER-TN: Averaged Results over {n_repititions} repititions')

        for learning_rate in learning_rates:

            print(f'LEARNING RATE: {learning_rate}')

            for policy in policies:

                print(f'POLICY: {policy}')

                if policy == 'egreedy':
                    print('skip')

                    #rewards_of_run_experiments = np.empty([n_repititions, n_episodes])
                    #for rep in range(n_repititions):

                    #    all_rewards_of_run = test_dqn_agent(network_method=network_method,
                    #                                        n_timesteps=n_timesteps,
                    #                                        n_episodes=n_episodes,
                    #                                        policy=policy,
                    #                                        tau=None,
                    #                                        learning_rate=learning_rate)

                    #    print(f'Rewards of experiment {rep + 1} for {policy}-annealing policy and alpha={learning_rate}: {all_rewards_of_run}')
                     #   rewards_of_run_experiments[rep] = all_rewards_of_run

                    #learning_curve = smooth(np.mean(rewards_of_run_experiments, axis=0), smoothing_window)  # average over repetitions
                    #MultipleRunPlot.add_curve(y=learning_curve, label=f'{policy}-annealing policy with alpha={learning_rate}')

                if policy == 'softmax':

                    for tau in taus:

                        rewards_of_run_experiments = np.empty([n_repititions, n_episodes])
                        for rep in range(n_repititions):

                            all_rewards_of_run = test_dqn_agent(target_network=target_network,
                                                                n_timesteps=n_timesteps,
                                                                n_episodes=n_episodes,
                                                                policy=policy,
                                                                tau=tau,
                                                                learning_rate=learning_rate)

                            print(f'Rewards of experiment {rep + 1} for {policy}-policy with tau={tau} and alpha={learning_rate}: {all_rewards_of_run}')
                            rewards_of_run_experiments[rep] = all_rewards_of_run

                        learning_curve = smooth(np.mean(rewards_of_run_experiments, axis=0), smoothing_window)  # average over repetitions
                        MultipleRunPlot.add_curve(y=learning_curve, label=f'{policy} policy with tau={tau} and alpha={learning_rate}')


        MultipleRunPlot.save('dqn_er_tn_learning_methods.png')

    else:
        print('Run experiment on DQN agent')


if __name__ == '__main__':
    # https://docs.python.org/3/library/argparse.html#module-argparse
    parser = argparse.ArgumentParser(prog='RLA2', description='DQN Experiments')

    # By default, the DQN does not have experience replay and target network
    parser.add_argument('--all_variations', action='store_true', dest='all_variations',
                        help='Comparison between all DQN variations.')
    parser.add_argument('--experience_replay', action='store_true', dest='experience_replay',
                        help='Add to the DQN agent experience replay (This argument will be ignored when --all_experiments is parsed).')
    parser.add_argument('--target_network', action='store_true', dest='target_network',
                        help='Add to the DQN agent an additional target network (This argument will be ignored when --all_experiments is parsed).')

    args = parser.parse_args()
    print(args)

    determine_experiment(args.all_variations, args.experience_replay, args.target_network)