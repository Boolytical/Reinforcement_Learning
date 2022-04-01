import argparse
from plotter import LearningCurvePlot
from dqn import act_in_env
import numpy as np
from plotter import smooth
import time
import concurrent.futures

# Test the dqn agent with given parameter values
def test_dqn_agent(NN, batch_size, target_network: str, n_timesteps, n_episodes, tau, learning_rate, policy, epsilon_decay, gamma):
    # The episode terminates if (pole angle greater than -12/12) or (cart position greater than -2.4,2.4) or (episode length exceeds 500)
    # Goal: Keep up the pole for 500 timesteps (as long as possible), if done=True too soon, then reward should be negative?

    param_dict = {
        'alpha': learning_rate,  # learning rate
        'gamma': gamma,  # discount factor
        'policy': policy,  # exploration strategy
        'epsilon': 1.0,  # initially, the agent will always choose a random action (exploration)
        'epsilon_min': 0.001,  # minimum value of exploration
        'epsilon_decay_rate': epsilon_decay,  # exploration behavior is gradually replaced by exploitation behavior
        'tau': tau,  # for softmax exploration strategy
        'max_replays': 2000,  # only a given amount of memory instants can be saved
        'NN' : NN,
        'batch_size': batch_size,  # number of samples from the memory that is used to fit the dnn model
        'target_network': target_network,  # has a target network (True) or not (False)
    }
    return act_in_env(n_episodes=n_episodes, n_timesteps=n_timesteps, param_dict=param_dict)


def run_egreedy(param_dic_run):
    policy = 'egreedy'
    rewards_of_run_experiments = np.empty([param_dic_run['n_repetitions'], param_dic_run['n_episodes']])

    for rep in range(param_dic_run['n_repetitions']):
        all_rewards_of_run = test_dqn_agent(NN=param_dic_run['NN'],
                                            batch_size=param_dic_run['batch_size'],
                                            target_network=param_dic_run['target_network'],
                                            n_timesteps=param_dic_run['n_timesteps'],
                                            n_episodes=param_dic_run['n_episodes'],
                                            policy=policy,
                                            tau=None,
                                            gamma = param_dic_run['gamma'],
                                            learning_rate=param_dic_run['learning_rate'],
                                            epsilon_decay=param_dic_run['decay_rate'])

        print('Rewards of one {}-annealing experiment with alpha={}, decay_rate={} and gamma={}:\n {}'.format(
            policy, param_dic_run["learning_rate"], param_dic_run["decay_rate"], param_dic_run["gamma"], all_rewards_of_run
        ))
        rewards_of_run_experiments[rep] = all_rewards_of_run
    return rewards_of_run_experiments


def run_softmax(param_dic_run):
    policy = 'softmax'
    rewards_of_run_experiments = np.empty([param_dic_run['n_repetitions'], param_dic_run['n_episodes']])

    for rep in range(param_dic_run['n_repetitions']):
        all_rewards_of_run = test_dqn_agent(NN=param_dic_run['NN'],
                                            batch_size=param_dic_run['batch_size'],
                                            target_network=param_dic_run['target_network'],
                                            n_timesteps=param_dic_run['n_timesteps'],
                                            n_episodes=param_dic_run['n_episodes'],
                                            policy=policy,
                                            gamma=param_dic_run['gamma'],
                                            tau=param_dic_run['tau'],
                                            learning_rate=param_dic_run['learning_rate'],
                                            epsilon_decay=None)

        print('Rewards of one {}-policy experiment with alpha={}, tau={} and gamma={}:\n {}'.format(
            policy, param_dic_run["learning_rate"], param_dic_run["tau"], param_dic_run["gamma"], all_rewards_of_run
        ))
        rewards_of_run_experiments[rep] = all_rewards_of_run
    return rewards_of_run_experiments


# Determine which DQN agent is used with experiment
def determine_experiment(all_variations: bool, experience_replay: bool, target_network: bool):
    n_episodes = 200  # number of episodes the agent will go through
    n_timesteps = 500  # number of timesteps one episode can maximally contain
    n_repetitions = 12 # number of repetitions per experiment setting
    n_processes = 6 # number of process to run in parallel
    reps_per_process = int(n_repetitions / n_processes) # repetitions of one experiment performed by one process
    smoothing_window = 51

    learning_rates = [0.05, 0.1, 0.2]
    epsilon_decay_rates = [0.005, 0.01, 0.02]
    taus = [0.1, 0.5, 1.0]
    gamma = 0.99
    NN = [64, 32]

    # For experiments below, all parameter experiments are performed
    if not experience_replay and not target_network:
        print('Run experiment on DQN agent')
        batch_size = 1
        target_network = False
        title = 'DQN'

    elif experience_replay and not target_network:
        print('Run experiment on DQN-ER agent')
        batch_size = 64
        target_network = False
        title = 'DQN-ER'

    elif target_network and not experience_replay:
        print('Run experiment on DQN-TN agent')
        batch_size = 1
        target_network = True
        title = 'DQN-TN'

    elif experience_replay and target_network:
        print('Run experiment on DQN-ER-TN agent')
        batch_size = 64
        target_network = True
        title = 'DQN-ER-TN'

        # #### Experiment 1: e-greedy
        # policy = 'egreedy'
        # MultipleRunPlot = LearningCurvePlot(title=f'{title} with {policy} annealing. Averaged Results over {n_repetitions} repetitions')
        #
        # for learning_rate in learning_rates:
        #     for decay_rate in epsilon_decay_rates:
        #         print(r'$\epsilon$-GREEDY APPROACH WITH FOLLOWING PARAMATER SETTINGS: $\alpha$={} and $\epsilon$-decay-rate={}'.format(learning_rate, decay_rate))
        #
        #         # Define list of dictionaries for each process
        #         # One dictionary contains parameters needed for running e-greedy function
        #         param_dics = []
        #         for _ in range(n_processes):
        #             param_dics.append({'learning_rate': learning_rate,
        #                                 'decay_rate': decay_rate,
        #                                 'gamma': gamma,
        #                                 'n_repetitions': reps_per_process,
        #                                 'n_episodes': n_episodes,
        #                                 'n_timesteps': n_timesteps,
        #                                 'target_network': target_network,
        #                                 'batch_size' : batch_size,
        #                                 'NN' : NN})
        #
        #         # Initialize array which will contain rewards of all repetitions for one setting
        #         rewards_of_run_experiments_all = np.empty([n_repetitions, n_episodes])
        #
        #         with concurrent.futures.ProcessPoolExecutor() as executor:
        #             # Map function: Run egreedy function with each parameter dictionary in param_dics
        #             # One mapping is one process!
        #             # Processes are run in parallel and map returns results and stores them in results_process
        #             results_process = executor.map(run_egreedy, param_dics)
        #
        #             # iterate through results of processes and combine them.
        #             # This is done by collecting all results in a global result array: rewards_of_run_experiments_all
        #             for i, result in enumerate(results_process):
        #
        #                 tmp = i * reps_per_process # help variable to store rewards of each process properly
        #                 rewards_of_run_experiments_all[tmp : tmp + reps_per_process, :] = result
        #
        #             # average over repetitions and smooth learning curve
        #             learning_curve = smooth(np.mean(rewards_of_run_experiments_all, axis=0), smoothing_window)
        #             MultipleRunPlot.add_curve(y=learning_curve,
        #                                       label=r'$\epsilon$-greedy with $\alpha$={} and $\epsilon$-decay-rate={}'.format(
        #                                           learning_rate, decay_rate))
        # MultipleRunPlot.save(f'{title}_learning_methods_{policy}_different_settings.png')
        #
        #
        # #### Experiment 2: Softmax
        # policy = 'softmax'
        # MultipleRunPlot = LearningCurvePlot(title=f'{title} with {policy} policy. Averaged Results over {n_repetitions} repetitions')
        #
        # for learning_rate in learning_rates:
        #     for tau in taus:
        #         print(r'SOFTMAX APPROACH WITH FOLLOWING PARAMATER SETTINGS: $\alpha$={} and $\tau$={}'.format(
        #                 learning_rate, tau))
        #
        #         # Define list of dictionaries for each process
        #         # One dictionary contains parameters needed for running softmax function
        #         param_dics = []
        #         for _ in range(n_processes):
        #             param_dics.append({'learning_rate': learning_rate,
        #                                'tau': tau,
        #                                'gamma': gamma,
        #                                'n_repetitions': reps_per_process,
        #                                'n_episodes': n_episodes,
        #                                'n_timesteps': n_timesteps,
        #                                'target_network': target_network,
        #                                'batch_size' : batch_size,
        #                                'NN' : NN}
        #                               )
        #         rewards_of_run_experiments_all = np.empty([n_repetitions, n_episodes])
        #
        #         with concurrent.futures.ProcessPoolExecutor() as executor:
        #             # Map function: Run softmax function with each parameter dictionary in param_dics
        #             results_process = executor.map(run_softmax, param_dics)
        #
        #             # iterate through results of processes and combine them.
        #             for i, result in enumerate(results_process):
        #                 tmp = i * reps_per_process # help variable to store rewards of each process properly
        #                 rewards_of_run_experiments_all[tmp: tmp + reps_per_process, :] = result
        #
        #             # average over repetitions and smooth learning curve
        #             learning_curve = smooth(np.mean(rewards_of_run_experiments_all, axis=0), smoothing_window)
        #             MultipleRunPlot.add_curve(y=learning_curve,
        #                                       label=r'{}-policy with $\alpha$={} and $\tau$-rate={}'.format(
        #                                           policy, learning_rate, tau))
        # MultipleRunPlot.save(f'{title}_learning_methods_{policy}_different_settings.png')
        #
        # #### Experiment 3: Tune Gamma-Parameter for best models
        # learning_rate, decay_rate, tau = 0.1, 0.001, 0.5 # Fix optimal parameters
        # gammas = [0.5, 0.75, 0.99]
        # MultipleRunPlot = LearningCurvePlot(title=r'Comparison of best {} models with different discount rate $\gamma$.' '\n'
        #                                           r'Averaged Results over {} repetitions'.format(title, n_repetitions))
        #
        # for policy in ('egreedy', 'softmax'):
        #     for gamma in gammas:
        #         print(
        #             'EGREEDY APPROACH WITH FOLLOWING PARAMATER SETTINGS: alpha={} and epsilon-decay={}\n'
        #             'SOFTMAX APPROACH WITH FOLLOWING PARAMATER SETTINGS: alpha={} and tau={}\n'
        #             'GAMMA PARAMETER $\gamma$ SET TO {}'.format(
        #                 learning_rate, decay_rate, learning_rate, tau, gamma))
        #
        #         if policy == 'egreedy':
        #             # Define list of dictionaries for each process
        #             # One dictionary contains parameters needed for running e-greedy function
        #             param_dics = []
        #             for _ in range(n_processes):
        #                 param_dics.append({'learning_rate': learning_rate,
        #                                    'batch_size' : batch_size,
        #                                    'decay_rate': decay_rate,
        #                                    'gamma': gamma,
        #                                    'n_repetitions': reps_per_process,
        #                                    'n_episodes': n_episodes,
        #                                    'n_timesteps': n_timesteps,
        #                                    'target_network': target_network,
        #                                    'NN' : NN}
        #                                   )
        #             rewards_of_run_experiments_all = np.empty([n_repetitions, n_episodes])
        #
        #             with concurrent.futures.ProcessPoolExecutor() as executor:
        #                 # Map function: Run egreedy function with each parameter dictionary in param_dics
        #                 results_process = executor.map(run_egreedy, param_dics)
        #
        #                 # Iterate through results of processes and combine them.
        #                 for i, result in enumerate(results_process):
        #                     tmp = i * reps_per_process  # help variable to store rewards of each process properly
        #                     rewards_of_run_experiments_all[tmp: tmp + reps_per_process, :] = result
        #
        #                 # Average over repetitions and smooth learning curve
        #                 learning_curve = smooth(np.mean(rewards_of_run_experiments_all, axis=0), smoothing_window)
        #                 MultipleRunPlot.add_curve(y=learning_curve,
        #                                           label=r'$\epsilon$-greedy with $\gamma$={}'.format(gamma))
        #
        #         if policy == 'softmax':
        #             # Define list of dictionaries for each process
        #             # One dictionary contains parameters needed for running softmax function
        #             param_dics = []
        #             for _ in range(n_processes):
        #                 param_dics.append({'learning_rate': learning_rate,
        #                                    'tau': tau,
        #                                    'gamma': gamma,
        #                                    'n_repetitions': reps_per_process,
        #                                    'n_episodes': n_episodes,
        #                                    'n_timesteps': n_timesteps,
        #                                    'target_network': target_network,
        #                                    'NN' : NN}
        #                                   )
        #
        #             rewards_of_run_experiments_all = np.empty([n_repetitions, n_episodes])
        #             with concurrent.futures.ProcessPoolExecutor() as executor:
        #                 # Map function: Run softmax function with each parameter dictionary in param_dics
        #                 results_process = executor.map(run_softmax, param_dics)
        #
        #                 # iterate through results of processes and combine them.
        #                 for i, result in enumerate(results_process):
        #                     tmp = i * reps_per_process  # help variable to store rewards of each process properly
        #                     rewards_of_run_experiments_all[tmp: tmp + reps_per_process, :] = result
        #
        #                 # average over repetitions and smooth learning curve
        #                 learning_curve = smooth(np.mean(rewards_of_run_experiments_all, axis=0), smoothing_window)
        #                 MultipleRunPlot.add_curve(y=learning_curve,
        #                                           label=r'Softmax policy with $\gamma$={}'.format(gamma))
        # MultipleRunPlot.save(f'optimal_dqn_er_tn_learning_models_different_gammas.png')
        #
        #### Experiment 4: network architecture
        NN = [[24, 24], [64, 32], [24, 24, 24]]
        policies = ['egreedy', 'softmax']
        optimal_gamma = 0.99

        for policy in policies:
            for architecture in NN:
                MultipleRunPlot = LearningCurvePlot(title=f'{title} with {policy} annealing. Averaged Results over {n_repetitions} repetitions')

                if policy == 'egreedy':
                    print(r'$\epsilon$-GREEDY APPROACH WITH FOLLOWING PARAMATER SETTINGS: $\alpha$={} and $\epsilon$-decay-rate={}'.format(
                        learning_rate, decay_rate))
                elif policy == 'softmax':
                    print(r'SOFTMAX APPROACH WITH FOLLOWING PARAMATER SETTINGS: $\alpha$={} and $\tau$={}'.format(
                        learning_rate, tau))

                # Define list of dictionaries for each process
                # One dictionary contains parameters needed for running softmax function
                param_dics = []
                for _ in range(n_processes):
                    param_dics.append({'NN' : architecture,
                                       'learning_rate': learning_rate,
                                       'decay_rate': decay_rate,
                                       'tau': tau,
                                       'gamma': optimal_gamma,
                                       'n_repetitions': reps_per_process,
                                       'n_episodes': n_episodes,
                                       'n_timesteps': n_timesteps,
                                       'target_network': target_network,
                                       'batch_size': batch_size}
                                      )
                rewards_of_run_experiments_all = np.empty([n_repetitions, n_episodes])

                with concurrent.futures.ProcessPoolExecutor() as executor:
                    # Map function: Run softmax function with each parameter dictionary in param_dics
                    if policy == 'egreedy':
                        results_process = executor.map(run_egreedy, param_dics)
                    elif policy == 'softmax':
                        results_process = executor.map(run_softmax, param_dics)

                    # iterate through results of processes and combine them.
                    for i, result in enumerate(results_process):
                        tmp = i * reps_per_process  # help variable to store rewards of each process properly
                        rewards_of_run_experiments_all[tmp: tmp + reps_per_process, :] = result

                    # average over repetitions and smooth learning curve
                    learning_curve = smooth(np.mean(rewards_of_run_experiments_all, axis=0), smoothing_window)
                    if policy == 'egreedy':
                        MultipleRunPlot.add_curve(y=learning_curve,
                                                  label=r'$\epsilon$-greedy with $\alpha$={} and $\epsilon$-decay-rate={}'.format(
                                                      learning_rate, decay_rate))
                    elif policy == 'softmax':
                        MultipleRunPlot.add_curve(y=learning_curve,
                                                  label=r'{}-policy with $\alpha$={} and $\tau$-rate={}'.format(
                                                      policy, learning_rate, tau))
            MultipleRunPlot.save(f'{title}_learning_methods_{policy}_different_settings.png')


    ## TO DO: implement the final check between all DQN tests based on optimal DQN-ER-TN parameters

def main():
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


if __name__ == '__main__':
    start = time.time()
    main()
    print('Total Run takes {} minutes'.format((time.time() - start) / 60))
