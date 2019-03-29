from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
import random, time
from collections import deque


"""

I think it works, but there is still a lot I don't fully understand. I want to have
a list of parameters and hyperparameters here and a full description of what they do
and what effect they have and why we are choosing this value. This includes activation
functions and things like that.

Also, expressively comment the main code like in the replay() and learn() methods

"""
class DQN_Learner:
    def __init__(self, sim):
        # the environment / simulation
        self.sim = sim
        # input dimensions / state size
        self.state_size = False
        # output dimensions / action size
        self.action_size = self.sim.action_space.n
        # list of rewards over episodes
        self.rewards = list()
        # to print the map if we get a record score
        self.best_reward = -99999999
        # exploration rate, decays over time
        self.max_eps = 1.0
        self.min_eps = 0.01
        self.eps_decay_rate = 0.001
        self.eps = self.max_eps
        # delayed reward factor / discount rate
        self.gamma = 0.99
        # learning rate
        self.alpha = 0.001
        # reset some parameters to simulation specific values
        self.set_params()
        # the neural network
        self.model = self._make_model()
        # memory stack for experience replay
        self.memory = deque(maxlen=2000)

    # reset the NN, rewards, and simulation
    def reset(self):
        self.rewards = list()
        self.best_reward = -99999999
        self.sim.reset()
        self.set_params()
        self.model = self._make_model()

    # set appropriate variables depending on the environment
    def set_params(self):
        if self.sim.spec.id == "FrozenLake-v0":
            self.max_eps = 1.0
            self.min_eps = 0.01
            self.eps_decay_rate = 0.001
            self.eps = self.max_eps
            self.gamma = 0.99
            self.alpha = 0.001
            self.state_size = 1
        elif self.sim.spec.id == "CartPole-v0":
            self.max_eps = 0.7
            self.min_eps = 0.01
            self.eps_decay_rate = 0.001
            self.eps = self.max_eps
            self.gamma = 0.99
            self.alpha = 0.001
            self.state_size = self.sim.observation_space.shape[0]
        elif self.sim.spec.id == "gym-forestfire-v0":
            self.max_eps = 1.0
            self.min_eps = 0.01
            self.eps_decay_rate = 0.005
            self.eps = self.max_eps
            self.gamma = 0.99
            self.alpha = 0.01
            self.state_size = self.sim.observation_space.shape[0]

    # create the neural net
    def _make_model(self):
        layers = [
            # First layer: #_neurons = state_size (implicit)
            # Second layer (first hidden layer): 50 neurons
            Dense(units=24,
                  activation='relu',
                  input_dim=self.state_size),
            Dense(units=24,
                  activation='relu'),
            # Third layer (output layer): #_neurons = action_size
            Dense(units=self.action_size, 
                  activation='linear')
        ]
        model = Sequential(layers)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.alpha))
        return model

    def remember(self, state, action, reward, sprime, done):
        self.memory.append((state, action, reward, sprime, done))

    def choose_action(self, state, eps=None):
        # epsilon is either fixed, or passed as argument
        eps_threshold = self.eps if eps is None else eps
        # exploitation vs exploration
        if random.uniform(0, 1) > eps_threshold:
            return np.argmax(self.model.predict(state)[0])
        else:
            return self.sim.action_space.sample()

    # decay epsilon (slower rate of decay for higher episode_num)
    def decay_epsilon(self, episode_num):
        self.eps = self.min_eps + \
            (self.max_eps - self.min_eps) * \
            np.exp(-self.eps_decay_rate * episode_num)

    def replay(self, batch_size):
        # take a random sample of size batch_size from memory
        minibatch = random.sample(self.memory, batch_size)
        # for each memory in the sample
        for state, action, reward, sprime, done in minibatch:
            # update the weights according to the loss
            target = reward
            if not done:
                # target = reward + decay_rate * max_q_value(next_state)
                target = reward + self.gamma * \
                        np.amax(self.model.predict(sprime)[0])
            # future_target = max_??
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def learn(self, n_episodes=1000, batch_size=10):
        for episode in range(n_episodes):
            done = False
            state = self.sim.reset()
            state = np.reshape(state, [1, self.state_size])
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                sprime, reward, done, _ = self.sim.step(action)
                if done:
                    break
                total_reward += reward
                sprime = np.reshape(sprime, [1, self.state_size])
                self.remember(state, action, reward, sprime, done)
                state = sprime

                if len(self.memory) > batch_size:
                    self.replay(batch_size)
                    # decay the exploration rate
                    self.decay_epsilon(episode)

            if total_reward > self.best_reward:
                self.best_reward = total_reward
                if self.sim.spec.id == "gym-forestfire-v0":
                    self.sim.render()

            print(f"Episode {episode + 1}: Total Reward --> {total_reward}")
            print(f"e: {self.eps}")
            self.rewards.append(total_reward)

    # play the simulation by choosing optimal Q-Table actions
    def play_optimal(self, eps=0):
        done = False
        state = self.sim.reset()
        while not done:
            self.sim.render()
            state = np.reshape(state, [1, self.state_size])
            action = self.choose_action(state, eps=eps)
            state, _, done, _ = self.sim.step(action)
            time.sleep(0.1)

    # plot the rewards against the episodes
    def show_rewards(self):
        plt.plot(self.rewards)
        plt.ylabel("Reward Values")
        plt.xlabel("Episodes")
        plt.show()

    # prints the average reward per k episodes
    def average_reward_per_k_episodes(self, k):
        n_episodes = len(self.rewards)
        rewards_per_k = np.split(np.array(self.rewards), n_episodes/k)
        count = k
        print(f"Average reward per {k} episodes:")
        for r in rewards_per_k:
            print(count, ":", str(sum(r/k)))
            count += k
