import gym
import numpy as np
import random
from tensorflow import keras
import matplotlib.pyplot as plt
from Helper import softmax

# ---------- Deep Q-Learning Agent ----------
class DQNAgent:
    
    def __init__(self, alpha, gamma, epsilon_start, epsilon_end, epsilon_decay_rate, tau, max_replays, batch_size):
        # (Hyper)parameters: (used values from https://towardsdatascience.com/deep-q-networks-theory-and-implementation-37543f60dd67) 
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor

        # Parameters of epsilon greedy policy
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end # Initially, the agent will always choose a random action (exploration)
        self.epsilon_decay_rate = epsilon_decay_rate # The exploration behavior is gradually replaced by exploitation behavior
        self.parameter = epsilon_end
        self.steps = 0 # Keep track of the decay steps

        # Parameters of softmax policy
        self.tau = tau # level of exploration
        self.action_selected = dict()

        self.max_replays = max_replays # Only a given amount of memory instants can be saved
        self.batch_size = batch_size # Number of samples from the memory that is used to fit the dnn model
        
        self.n_states = 4 # Number of used state parameters, Actions: (0) Push cart to left, (1) Push cart to right
        self.n_actions = 2 # Number of possible actions
        # Observation Space: [0] Cart position (-4.8, 4.8), [1] cart velocity (-Inf, Inf), [2] pole angle (-24, 24), [3] pole angular velocity (-Inf, Inf)
        
        self.replay_memory = [] # Store the experience of the agent in tuple (s_t, a_t, r_t+1, s_t+1)
        
        self.dnn_model = self._initialize_dnn()
        self.dnn_model_static = self._initialize_dnn() # Use this model for generating the target value
        
    # Build the dnn model that outputs the Q-values for every action given a specific state 
    def _initialize_dnn(self):
        print('Create a neural network with {} input nodes and {} output nodes'.format(self.n_states, self.n_actions))

        #initializer = keras.initializers.Zeros() Set all weights to zero?

        model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.n_states,)),
                keras.layers.Dense(units=24, activation='relu'),
                keras.layers.Dense(units=24, activation='relu'),
                keras.layers.Dense(units=self.n_actions, activation='linear')
            ]
        )
        
        model.summary()
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.alpha))
        
        return model # Return the initial dnn model
    
    # Memorize given experience
    def memorize(self, s: list, a: int, r: float, s_next: list, done: bool):        
        self.replay_memory.append((s, a, r, s_next, done))
        
        if len(self.replay_memory) > self.max_replays: # If memory size exceeds the set limit
            self.replay_memory.pop(0) # Remove the oldest 
            
    # Improve the model by feeding a batch of samples from the saved memory
    def learn(self):
        if len(self.replay_memory) >= self.batch_size: # Only sample and learn if the agent has collected enough experience
    
            batch_memory = random.sample(self.replay_memory, self.batch_size) # Every memory can be selected only once
            
            current_weights = self.dnn_model.get_weights() # Get the current weights of the model
            self.dnn_model_static.set_weights(current_weights) # Freeze the current model into the static model
            
            for s, a, r, s_next, done in batch_memory:
                
                if done:
                    target = r
                else:
                    q_values_of_s_next = self.dnn_model_static.predict(s_next)
                    target = r + self.gamma * np.amax(q_values_of_s_next)
                    
                targets_fit = self.dnn_model_static.predict(s) # Get the predicted target values 
                targets_fit[0][a] = target # Insert the calculated Q-value 
                
                # Use the current model again to fit the weights
                self.dnn_model.fit(s, targets_fit, verbose=0) # Fit the model to the new target values given s
                    
    # Choose action combined with epsilon greedy method to balance between exploration and exploitation
    def choose_action(self, s, policy):
        if policy == 'egreedy':
            # Decay the epsilon greedy
            self.parameter = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1 * self.steps / self.epsilon_decay_rate)
            self.steps += 1

            if np.random.uniform(0, 1) > self.parameter:
                a = np.argmax(self.dnn_model.predict(s)) # Choose action with highest Q-value
            else:
                a = env.action_space.sample() # Choose random action
        elif policy == 'softmax':
            prob = softmax(self.dnn_model.predict(s), self.tau)
            a = np.argmax(prob)

        return a # Return chosen action
    


# --------- https://gym.openai.com/docs/ ----------
def DQN(n_episodes, n_timesteps, alpha, gamma, policy, epsilon_start, epsilon_end, epsilon_decay_rate, tau, max_replays, batch_size):
    # Create environment of CartPole-v1
    env = gym.make('CartPole-v1')
    dqn_agent = DQNAgent(alpha, gamma, epsilon_start, epsilon_end, epsilon_decay_rate, tau, max_replays, batch_size)

    e_scores = []
    for e in range(n_episodes):

        state = env.reset() # Reset environment
        state = np.array([state]) # Create model compatible shape

        for t in range(n_timesteps):
            env.render()

            action = dqn_agent.choose_action(state, t, policy) # Choose an action given the current state
            state_next, reward, done, _ = env.step(action) # Last variable never used
            state_next = np.array([state_next]) # Create model compatible shape

            if done and t < 500: # Will it return done=True when > 500?
                reward = -10 # Give negative reward when done before maximum number of timesteps reached Does this matter?

            dqn_agent.memorize(state, action, reward, state_next, done) # Include this experience to the memory

            state = state_next

            if done:
                print("Episode {} finished after {} timesteps".format(e+1, t+1))
                e_scores.append(t+1)
                break

        dqn_agent.learn() # Learn from current collected experience
        dqn_agent.decrease_epsilon() # Decrease exploration probability after every episode
        print("Parameter epsilon decreased to {}".format(dqn_agent.epsilon))
    env.close()

    return e_scores



def test():
    # The episode terminates if (pole angle greater than -12/12) or (cart position greater than -2.4,2.4) or (episode length exceeds 500)
    # Goal: Keep up the pole for 500 timesteps (as long as possible), if done=True too soon, then reward should be negative?
    n_episodes = 1000 # Number of episodes the agent will go through
    n_timesteps = 500 # Number of timesteps one episode can maximally contain

    alpha = 0.001  # learning rate
    gamma = 0.99  # discount factor

    policy = 'egreedy' # 'egreedy' or 'UCB'
    epsilon_start = 0.001
    epsilon_end = 1.0  # Initially, the agent will always choose a random action (exploration)
    epsilon_decay_rate = 0.001  # The exploration behavior is gradually replaced by exploitation behavior
    tau = 0.5

    max_replays = 2000  # Only a given amount of memory instants can be saved
    batch_size = 32  # Number of samples from the memory that is used to fit the dnn model

    e_scores = DQN(n_episodes, n_timesteps, alpha, gamma, policy, epsilon_start, epsilon_end, epsilon_decay_rate, tau, max_replays, batch_size)

    # Plot the e-scores
    plt.plot(range(n_episodes), e_scores)
    plt.xlabel('Number of episodes')
    plt.ylabel('Duration')
    plt.show()



if __name__ == '__main__':
    test()