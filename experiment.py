import numpy as np
import time
from dqn import DQN
from Helper import LearningCurvePlot, smooth

def average_over_repetitions(n_repetitions, n_episodes, n_timesteps, alpha, gamma, policy, epsilon_start, epsilon_end, epsilon_decay_rate,
                             tau, max_replays, batch_size, smoothing_window=51):

    e_scores_results = np.empty([n_episodes, n_timesteps])  # Result array
    now = time.time()

    for rep in range(n_repetitions):  # Loop over repetitions
        e_scores = DQN(n_episodes, n_timesteps, alpha, gamma, policy, epsilon_start, epsilon_end,
                       epsilon_decay_rate, tau, max_replays, batch_size)

        e_scores_results[rep] = e_scores

    print('Running one setting takes {} minutes'.format((time.time() - now) / 60))
    learning_curve = np.mean(e_scores_results, axis=0)  # average over repetitions
    learning_curve = smooth(learning_curve, smoothing_window)  # additional smoothing
    return learning_curve



def experiment():
    # Experiment
    n_repetitions = 50
    smoothing_window = 1001


    ## DQN parameters ##
    n_episodes = 1000 # Number of episodes the agent will go through
    n_timesteps = 500 # Number of timesteps one episode can maximally contain

    alpha = 0.001  # learning rate
    gamma = 0.99  # discount factor

    policy = 'egreedy' # 'egreedy' or 'softmax'
    epsilon_start = 0.001
    epsilon_end = 1.0  # Initially, the agent will always choose a random action (exploration)
    epsilon_decay_rate = 0.001  # The exploration behavior is gradually replaced by exploitation behavior
    tau = 0.5

    max_replays = 2000  # Only a given amount of memory instants can be saved
    batch_size = 32  # Number of samples from the memory that is used to fit the dnn model


    ## Experiment of DQN ##
    # Test 1: optimal epsilon greedy parameters
    Plot = LearningCurvePlot(title = 'DQN: parameter tuning of $\epsilon$-greedy exploration')
    policy = 'egreedy'
    epsilons_end = [1.0, 0.75, 0.5]
    epsilons_decay = [0.001, 0.05, 0.1]
    for end in epsilons_end:
        for decay in epsilons_decay:
            learning_curve = average_over_repetitions(n_repetitions, n_episodes, n_timesteps, alpha, gamma, policy,
                                                  epsilon_start, end, decay, tau, max_replays, batch_size)
            Plot.add_curve(learning_curve,label=r'$\epsilon$-greedy, $\epsilon $ = {}, decay = {}'.format(end, decay))
    Plot.save('optimal_egreedy.png')
    ## TO DO: after optimal epsilon_end and epsilon_decay_rate are determined, set them at the start for the other experiments! ##


    # Test 2: optimal epsilon-greedy vs softmax
    Plot = LearningCurvePlot(title = 'DQN: effect of $\epsilon$-greedy versus softmax exploration')
    policy = 'egreedy'
    learning_curve = average_over_repetitions(n_repetitions, n_episodes, n_timesteps, alpha, gamma, policy, epsilon_start,
                                              tau, epsilon_end, epsilon_decay_rate, tau, max_replays, batch_size)
    Plot.add_curve(learning_curve,label=r'$\epsilon$-greedy, $\epsilon $ = {}, decay = {}'.format(epsilon_end, epsilon_decay_rate))

    policy = 'softmax'
    taus = [0.1, 0.5, 1.0]
    for C in taus:
        learning_curve = average_over_repetitions(n_repetitions, n_episodes, n_timesteps, alpha, gamma, policy,
                                                  epsilon_start, epsilon_end, epsilon_end, C, max_replays, batch_size)
        Plot.add_curve(learning_curve,label=r'softmax, C = {}'.format(C))
    Plot.save('egreedy_vs_softmax.png')
    ## TO DO: after optimal tau is determined, set it at the start for the other experiments! ##


    # Test 3: optimal learning and discount rate
    Plot = LearningCurvePlot(title = 'DQN: optimal learning and discount rate for ... policy') ## TO DO: change this policy to the optimal one ##
    policy = 'egreedy' ## TO DO: change this to the optimal policy ##
    alphas = [0.001, 0.01, 0.1]
    gammas = [1.0, 0.75, 0.5]
    for a in alphas:
        for g in gammas:
            learning_curve = average_over_repetitions(n_repetitions, n_episodes, n_timesteps, a, g, policy,
                                                  epsilon_start, epsilon_end, epsilon_decay_rate, tau, max_replays, batch_size)
        Plot.add_curve(learning_curve,label=r'$\epsilon$-greedy, $\alpha $ = {}, $\gamma $ = {}'.format(a, g)) ## TO DO: change policy ##
    Plot.save('learning_discount.png')

    ## TO DO: add tests to compare DQN with DQN-ER-TN ##



if __name__ == '__main__':
    experiment()