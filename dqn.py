import gym
import numpy as np
import random
from tensorflow import keras

from util import softmax


# ---------- Deep Q-Learning Agent ----------
class DQNAgent:
    
    def __init__(self, param_dict: dict):
        # (Hyper)parameters: (used values from https://towardsdatascience.com/deep-q-networks-theory-and-implementation-37543f60dd67) 
        self.alpha = param_dict['alpha']
        self.gamma = param_dict['gamma']

        self.policy = param_dict['policy']

        # Parameters of epsilon greedy policy
        self.epsilon = param_dict['epsilon']
        self.epsilon_min = param_dict['epsilon_min']
        self.epsilon_decay_rate = param_dict['epsilon_decay_rate']

        # Parameters of softmax policy
        self.tau = param_dict['tau']

        self.max_replays = param_dict['max_replays']
        self.batch_size = param_dict['batch_size']
        
        # Observation Space: [0] Cart position (-4.8, 4.8), [1] cart velocity (-Inf, Inf), [2] pole angle (-24, 24), [3] pole angular velocity (-Inf, Inf)
        self.n_states = 4   # number of used state parameters
        # Actions: (0) Push cart to left, (1) Push cart to right 
        self.n_actions = 2  # number of possible actions

        self.replay_memory = [] # Store the experience of the agent in tuple (s_t, a_t, r_t+1, s_t+1)
        
        self.dnn_model = self._initialize_dnn()

        if param_dict['target_network']:
            self.dnn_model_static = self._initialize_dnn() # Use this model for generating the target value
        else:
            self.dnn_model_static = None
        
    # Build the dnn model that outputs the Q-values for every action given a specific state 
    def _initialize_dnn(self):
        print('Create a neural network with {} input nodes and {} output nodes'.format(self.n_states, self.n_actions))

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.n_states,)),
                keras.layers.Dense(units=64, activation='relu'),
                keras.layers.Dense(units=32, activation='relu'),
                keras.layers.Dense(units=self.n_actions, activation='linear')
            ]
        )
        
        model.summary()
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.alpha))
        
        return model    # Return the dnn model
    
    # Memorize given experience
    def memorize(self, s: list, a: int, r: float, s_next: list, done: bool):        
        self.replay_memory.append((s, a, r, s_next, done))
        
        if len(self.replay_memory) > self.max_replays: # If memory size exceeds the set limit
            self.replay_memory.pop(0) # Remove the oldest 
            
    # Improve the model by feeding one per time of a batch of samples from the saved memory with or without target network
    def learn(self):
        if len(self.replay_memory) >= self.batch_size: # Only sample and learn if the agent has collected enough experience
    
            batch_memory = random.sample(self.replay_memory, self.batch_size) # Every memory can be selected only once
            
            if self.dnn_model_static:
                current_weights = self.dnn_model.get_weights() # Get the current weights of the model
                self.dnn_model_static.set_weights(current_weights) # Freeze the current model into the static model
            
            for s, a, r, s_next, done in batch_memory:
                
                if done:
                    target = r
                else:

                    if self.dnn_model_static:
                        q_values_of_s_next = self.dnn_model_static.predict(s_next)
                    else:
                        q_values_of_s_next = self.dnn_model.predict(s_next)

                    target = r + self.gamma * np.amax(q_values_of_s_next)
                    
                if self.dnn_model_static:
                    targets_fit = self.dnn_model_static.predict(s)  # Get the predicted target values
                else:
                    targets_fit = self.dnn_model.predict(s) # Get the predicted target values

                targets_fit[0][a] = target # Insert the calculated Q-value 
                
                # Use the current model again to fit the weights
                self.dnn_model.fit(s, targets_fit, verbose=0) # Fit the model to the new target values given s

    # Improve the model by feeding a batch of samples from the saved memory using one model only
    def learn_batch_wise(self):
        if len(self.replay_memory) >= self.batch_size:
            batch = random.sample(self.replay_memory, self.batch_size)
            states = np.zeros((self.batch_size, self.n_states))  # Dim: batch_size x 4
            states_next = np.zeros((self.batch_size, self.n_states))  # Dim: batch_size x 4
            actions, rewards, dones = [], [], []

            for cnt, experience in enumerate(batch):
                states[cnt, :] = experience[0] # Collect states of all experiences from batch
                states_next[cnt, :] = experience[3] # Collect new states of all experiences from batch
                actions.append(experience[1]) # Collect actions of all experiences from batch
                rewards.append(experience[2]) # Collect rewards of all experiences from batch
                dones.append(experience[4])

            output = self.dnn_model.predict(states)  # Dim: batch_size x 2(actions) --> Primary network
            target = self.dnn_model.predict(states_next)  # Dim: batch_size x 2(actions)

            for i in range(self.batch_size):
                if not dones[i]:
                    output[i, :][actions[i]] = rewards[i] + self.gamma * np.max(target[i])
                else:
                    output[i, :][actions[i]] = rewards[i]

            self.dnn_model.fit(states, output, verbose=0)
                    
    # Choose action combined with epsilon greedy method to balance between exploration and exploitation
    def choose_action(self, s):
        if self.policy == 'egreedy':
            # current_epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1 * self.steps / self.epsilon_decay_rate)
            # self.steps += 1

            if np.random.uniform(0, 1) > self.epsilon:
                a = np.argmax(self.dnn_model.predict(s)) # Choose action with highest Q-value
            else:
                a = np.random.randint(0,self.n_actions)   # Choose random action 
        
        elif self.policy == 'softmax':
            raise KeyError('Softmax not yet implemented')

        else:
            raise KeyError(f'Given policy {self.policy} not existing')

        return a # Return chosen action

    # Decay epsilon
    def decay_epsilon(self, n_episodes: int):
        if self.policy == 'egreedy' and self.epsilon > self.epsilon_min:
            epsilon_delta = (self.epsilon - self.epsilon_min) / n_episodes
            self.epsilon = self.epsilon - epsilon_delta   


def act_in_env(n_episodes: int, n_timesteps: int, param_dict: dict):

    env = gym.make('CartPole-v1')   # create environment of CartPole-v1
    dqn_agent = DQNAgent(param_dict)

    env_scores = []

    for e in range(n_episodes):

        state = env.reset() # reset environment and get initial state
        state = np.array([state])   # create model compatible shape

        for t in range(n_timesteps):
            # env.render()

            action = dqn_agent.choose_action(state)  # choose an action given the current state
            state_next, reward, done, _ = env.step(action)  # last variable never used
            state_next = np.array([state_next]) # create model compatible shape

            dqn_agent.memorize(state, action, reward, state_next, done) # include this experience to the memory

            state = state_next

            if done:
                print("Episode {} with epsilon {} finished after {} timesteps".format(e+1, dqn_agent.epsilon, t+1))
                env_scores.append(t+1)
                break
        
        if param_dict['learn_batch_wise']:
            dqn_agent.learn_batch_wise() # learn from current collected experience feeding whole batch to network
        else:
            dqn_agent.learn() # learn from current collected experience feeding one experience of batch to the network per time

        dqn_agent.decay_epsilon(n_episodes) # decay epsilon after every episode

    env.close()

    return env_scores
