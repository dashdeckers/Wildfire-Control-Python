import time, random
import numpy as np
import matplotlib.pyplot as plt

"""
> Q-Table approach:

All simple implementation use a pre-initialized matrix as a Q-Table,
but because we have a continuous state space we do not have a fixed
number of either columns or rows (the other is the action space).

To work around this, we have the following options (from easy to hard):
- Discretize the state space by rounding the distance and angle vars
- Implement a HashTable, which doesn't need to know the size beforehand
- Implement a DQN, which approximates the Q-Function with a Neural Net

Discretization and the HashTable should be used together, and should be
our first attempt. It will be less accurate (rounding), less optimized
than a matrix, and might use a lot of memory (inherent to the Q-Table
approach)

The DQN will use less memory and won't introduce inaccuracy via rounding,
but will be much more computationally expensive.


> Q-Learner (Q-Table approach):

The class implements Q-Learning with:
- A dynamic table (a dict / hashtable)
- Constant windspeed + direction
- Discretization of the feature/state space (rounding to 1 decimal)
- Reduced the features (removed the last 4 entries). 

The last two points are done in the hope that similar states collapse to 
give identical actions and get updated together. This really distorts the
agents view of the environment, however, and introduces a lot of inaccuracy

I didnt do a parameter sweep yet, so maybe we should do that.

Also, 1000 episodes are just not enough. Much more simple environments
(like frozenlake) need to run for 10,000 episodes before reaching decent
scores.

In general, this is not gonna cut it anyway. We need a DQN.
"""

class QT_Learner:
    def __init__(self, sim):
        # simulation
        self.sim = sim
        # preprocess states (lists to tuples)
        self.preprocess = True
        # q-table
        self.QT = dict()
        # list of rewards over episodes
        self.rewards = list()
        # to print the map if we get a record score
        self.best_reward = -99999999
        # chance of taking random action, decays over time
        self.max_eps = 1
        self.min_eps = 0.01
        self.eps_decay_rate = 0.001
        self.eps = self.max_eps
        # delayed reward factor
        self.gamma = 0.99
        # learning rate
        self.alpha = 0.1
        # set the hyper parameters
        self.set_params()

    # reset the Q-Table, rewards, and simulation
    def reset(self):
        self.QT = dict()
        self.rewards = list()
        self.best_reward = -99999999
        self.sim.reset()
        self.eps = self.max_eps

    # set appropriate variables depending on the environment
    def set_params(self):
        if self.sim.spec.id == "FrozenLake-v0":
            self.preprocess = False
            self.alpha = 0.1
            self.gamma = 0.99
            self.max_eps = 1
            self.min_eps = 0.01
            self.eps_decay_rate = 0.001
            self.eps = self.max_eps
        elif self.sim.spec.id == "gym-forestfire-v0":
            self.preprocess = True
            self.alpha = 0.2
            self.gamma = 0.99
            self.max_eps = 1
            self.min_eps = 0.2
            self.eps_decay_rate = 0.001
            self.eps = self.max_eps

    """
    A dynamic Q-Table: QT[state] = numpy.array(num_actions)

    Access the reward for a state, action pair: Q(s, a)
    by calling qtable(state, action).

    Get all actions and rewards via qtable(state)
    """
    def qtable(self, state, action=None):
        if self.preprocess:
            state = tuple(state)
        if state not in self.QT:
            self.QT[state] = np.zeros(self.sim.action_space.n)
        if action is None:
            return self.QT[state]
        return self.QT[state][action]

    # decay epsilon (slower rate of decay for higher episode_num)
    def decay_epsilon(self, episode_num):
        self.eps = self.min_eps + \
            (self.max_eps - self.min_eps) * \
            np.exp(-self.eps_decay_rate * episode_num)

    # choose an action via e-greedy method
    def choose_action(self, state, eps=None):
        if self.preprocess:
            state = tuple(state)
        # epsilon is either fixed, or passed as argument
        eps_threshold = self.eps if eps is None else eps
        # exploitation vs exploration
        if random.uniform(0, 1) > eps_threshold:
            return np.argmax(self.qtable(state))
        else:
            return self.sim.action_space.sample()

    """
    Q-Learning algorithm

    Until the simulation is over:

    Choose an action (randomly e percent of the time, optimally otherwise)
    Update the environment with that action
    Update the Q-Table entry for the previous state and action
    """
    def learn(self, n_episodes=1000):
        for episode in range(n_episodes):
            done = False
            state = self.sim.reset()
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                sprime, reward, done, _ = self.sim.step(action)
                total_reward += reward
                """
                self.qtable(state)[action] = self.qtable(state, action) + \
                        self.alpha * (reward + self.gamma * \
                        np.max(self.qtable(sprime)) - self.qtable(state, action))
                """
                self.qtable(state)[action] = (1 - self.alpha) * self.qtable(state, action) \
                        + self.alpha * (reward + self.gamma * np.max(self.qtable(sprime)))
                state = sprime

            self.decay_epsilon(episode)

            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.sim.render()

            print(f"Episode {episode + 1}: Total Reward --> {total_reward}")
            self.rewards.append(total_reward)

    # play the simulation by choosing optimal Q-Table actions
    def play_optimal(self, eps=0):
        done = False
        state = self.sim.reset()
        while not done:
            self.sim.render()
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
