from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from keras.utils import plot_model

import matplotlib.pyplot as plt
import numpy as np
import random, time, keras
from collections import deque


"""
This pretty much (apart from the TODO's) implements the DQN algorithm.

I want to have a list of parameters and hyper-parameters here and a
description of the effect they have, as well as the values they are set
to and why.
"""

# TODO: greyscale instead of one-hot-encoding
# TODO: 4-stacked frames
# TODO: 84x84 input (scale it)
# TODO: tune parameters
class DQN_Learner:
    def __init__(self, sim, small_network=False, name=None):
        if not sim.spec.id == "gym-forestfire-v0":
            print("DQN currently only supports ForestFire...")
            return
        # the environment / simulation
        self.sim = sim
        # output dimensions / action size
        self.action_size = self.sim.action_space.n
        # list of rewards over episodes
        self.rewards = list()
        # to print the map if we get a record score
        self.best_reward = -10000
        # exploration rate, decays over time
        self.max_eps = 1.0
        self.min_eps = 0.01
        self.eps_decay_rate = 0.005
        self.eps = self.max_eps
        # delayed reward factor / discount rate
        self.gamma = 0.99
        # learning rate
        self.alpha = 0.001
        # number of iterations before updating the target network
        self.target_update_cnt = 1000
        # the neural network
        self.model = self._make_model(small_network)
        # load a model from file if a name is given
        if name:
            self._load(name)
        # the target neural network
        self.target = keras.models.clone_model(self.model)
        self.target.set_weights(self.model.get_weights())
        # memory stack for experience replay
        # TODO: Pre-initialize memory with random-run data
        self.memory = deque(maxlen=100000)

    '''
    Create the neural net:

    layers_original is the replicated architecture from original DQN paper:
    input layer: WIDTH x HEIGHT x 5
    conv layer: 32 filters of 8x8 with stride 4 + ReLu
    conv layer: 64 filters of 4x4 with stride 2 + ReLu
    conv layer: 64 filters of 3x3 with stride 1 + ReLu
    dense layer: 512 units + ReLu
    dense layer: 6 units (output layer, 6 possible actions)

    layers_small is a smaller network with:
    conv layer: 32 filters of 8x8 with stride 4 + ReLu
    dense layer: 52 units + ReLu
    dense layer: 6 units (output layer, 6 possible actions)
    '''
    def _make_model(self, small_network=False):
        input_shape = (self.sim.width, self.sim.height, 5)
        layers_original = [
            Conv2D(filters=32,
                   kernel_size=(8, 8),
                   strides=4,
                   padding='same',
                   activation='relu',
                   data_format='channels_last',
                   input_shape=input_shape),
            Conv2D(filters=64,
                   kernel_size=(4, 4),
                   strides=2,
                   padding='same',
                   activation='relu'),
            Conv2D(filters=64,
                   kernel_size=(3, 3),
                   strides=1,
                   padding='same',
                   activation='relu'),
            Flatten(),
            Dense(units=512,
                  activation='relu'),
            Dense(units=self.action_size,
                  activation='linear')
        ]
        layers_small = [
            Conv2D(filters=32,
                   kernel_size=(5, 5),
                   strides=4,
                   padding='same',
                   activation='relu',
                   data_format='channels_last',
                   input_shape=input_shape),
            Flatten(),
            Dense(units=52,
                  activation='relu'),
            Dense(units=self.action_size,
                  activation='linear')
        ]
        if small_network:
            model = Sequential(layers_small)
        else:
            model = Sequential(layers_original)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.alpha))
        model.summary()
        return model

    # add a memory
    def _remember(self, state, action, reward, sprime, done):
        self.memory.append((state, action, reward, sprime, done))

    # choose action with e-greedy policy with current or given eps value
    def choose_action(self, state, eps=None):
        # epsilon is either fixed, or passed as argument
        eps_threshold = self.eps if eps is None else eps

        if random.uniform(0, 1) > eps_threshold:
            return np.argmax(self.model.predict(state)[0])
        else:
            return self.sim.action_space.sample()

    # decay epsilon (slower rate of decay for higher episode_num)
    # TODO: configure this to anneal to min_eps in exactly X episodes
    def _decay_epsilon(self, episode_num):
        self.eps = self.min_eps + \
            (self.max_eps - self.min_eps) * \
            np.exp(-self.eps_decay_rate * episode_num)

    # sample randomly from memory
    # TODO: implement error clipping
    # TODO: with 4-stacked memory, each element will be a list and i think
    # they take the cumulative reward for those stacks to fit with and predict
    # from the last state (??? thats already frame skipping ???)
    def _replay(self, batch_size):
        # take a random sample of size batch_size from memory
        minibatch = random.sample(self.memory, batch_size)
        # for each memory in the sample
        for state, action, reward, sprime, done in minibatch:
            # the target value is the estimated q-value the current state
            # should have, based on the q-values of the next state.
            # it is the reward if the current state is a terminal state
            target = reward
            # otherwise, it is calculated via the predicted value from the
            # target network on sprime, the discounting factor, and the reward
            if not done:
                # target = reward + discount_factor * max_q_value(next_state)
                target = reward + self.gamma * \
                        np.amax(self.target.predict(sprime)[0])
            # so the predicted value for the current state and action taken
            # should be more like the target value
            predicted = self.model.predict(state)
            predicted[0][action] = target
            self.model.fit(state, predicted, epochs=1, verbose=0)

            '''
            (target - predicted)**2

            # old syntax, from example
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            '''

    # the DQN algorithm. some of the algorithm is moved to the replay method
    # TODO: preinitialize, then always add a 4-stack of frames to memory.
    # This will have consequences for replay() as well
    def learn(self, n_episodes=1000, batch_size=32):
        target_update_counter = self.target_update_cnt
        for episode in range(n_episodes):
            t0 = time.time()
            total_reward = 0
            done = False
            # initialize state
            state = self.sim.reset()
            # model expects an array of samples, even if it is only one sample.
            # so state[0] should be the actual state, thats why the reshapes
            state = np.reshape(state, [1] + list(state.shape))

            while not done:
                # select action following e-greedy policy
                action = self.choose_action(state)
                # execute action and observe result
                sprime, reward, done, _ = self.sim.step(action)
                sprime = np.reshape(sprime, [1] + list(sprime.shape))
                # keep track of total reward
                total_reward += reward

                # store experience in replay memory
                self._remember(state, action, reward, sprime, done)
                # experience replay: learn from sampled memories
                if len(self.memory) > batch_size:
                    self._replay(batch_size)
                    # decay the exploration rate
                    self._decay_epsilon(episode)

                # every C iterations, update the target network
                target_update_counter -= 1
                if target_update_counter == 0:
                    target_update_counter = self.target_update_cnt
                    self.target.set_weights(self.model.get_weights())

                # set state to be the next state, for the next iteration
                state = sprime

            # render the last state if we reach a highscore
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.sim.render()

            print(f"Episode {episode + 1}: Total Reward --> {total_reward}")
            print(f"Epsilon: {self.eps}")
            print(f"Time taken: {time.time() - t0}\n")
            self.rewards.append(total_reward)

    # play the simulation by choosing optimal actions
    def play_optimal(self, eps=0):
        done = False
        state = self.sim.reset()
        while not done:
            self.sim.render()
            state = np.reshape(state, [1] + list(state.shape))
            action = self.choose_action(state, eps=eps)
            state, _, done, _ = self.sim.step(action)
            time.sleep(0.1)
        self.sim.render()

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

    # saves a visualization of the model to a file
    def show_model(self, name="model"):
        if name == "model":
            plot_model(self.model, show_shapes=True, to_file='model.png')
        if name == "target":
            plot_model(self.target, show_shapes=True, to_file='target.png')

    # loads the weights of the model from file.
    # pass name to class initialization to use load
    def _load(self, name):
        self.model.load_weights(name)

    # saves the weights of the model to file
    def save(self, name):
        self.model.save_weights(name)

