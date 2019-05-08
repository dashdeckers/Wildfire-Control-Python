from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from keras.utils import plot_model

import matplotlib.pyplot as plt
import numpy as np
import random, time, keras
import tensorflow as tf
from collections import deque
from Misc import Custom_TensorBoard

"""
This pretty much (apart from the TODO's) implements the DQN algorithm.

I want to have a list of parameters and hyper-parameters here and a
description of the effect they have, as well as the values they are set
to and why.

Error Clipping:
https://stackoverflow.com/questions/36462962/loss-clipping-in-tensor-flow-on-deepminds-dqn
AND
https://stackoverflow.com/questions/47840527/using-tensorflow-huber-loss-in-keras
"""

class DQN_Learner:
    def __init__(self, sim, name=None):
        if not sim.spec.id == "gym-forestfire-v0":
            print("DQN currently only supports ForestFire...")
            return
        # the environment / simulation
        self.sim = sim
        # the constants
        self.METADATA = sim.METADATA
        self.LOGGING = sim.LOGGING
        # output dimensions / action size
        self.action_size = self.sim.action_space.n
        # list of rewards over episodes
        self.rewards = list()
        # to print the map if we get a record score
        self.best_reward = -10000
        # exploration rate, decays over time
        self.max_eps = self.METADATA['max_eps']
        self.min_eps = self.METADATA['min_eps']
        self.eps_decay_rate = self.METADATA['eps_decay_rate']
        self.eps = self.max_eps
        # delayed reward factor / discount rate
        self.gamma = self.METADATA['gamma']
        # learning rate
        self.alpha = self.METADATA['alpha']
        # number of iterations before updating the target network
        self.target_update_cnt = self.METADATA['target_update']
        # the neural network
        self.model = self._make_model(self.sim.small_network)
        # load a model from file if a name is given
        if name:
            self._load(name)
        # the target neural network
        self.target = keras.models.clone_model(self.model)
        self.target.set_weights(self.model.get_weights())
        # memory stack for experience replay (do run_human() to pre-initialize it)
        self.memory = deque(maxlen=self.METADATA['memory_size'])

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
    dense layer: 52 units + ReLu
    dense layer: 6 units (output layer, 6 possible actions)
    '''
    def _make_model(self, small_network=False):
        if self.sim.FITNESS_MEASURE == "Toy":
            input_shape = (self.sim.W.WIDTH, self.sim.W.HEIGHT)
        else:
            input_shape = (self.sim.W.WIDTH, self.sim.W.HEIGHT, 3)
        layers_original = [
            Conv2D(filters=32,
                   kernel_size=(8, 8),
                   strides=4,
                   padding='same',
                   activation='relu',
                   data_format='channels_first',
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
            Flatten(input_shape=input_shape),
            Dense(units=52,
                  activation='sigmoid'),
            Dense(units=self.action_size,
                  activation='linear')
        ]
        if small_network:
            model = Sequential(layers_small)
        else:
            model = Sequential(layers_original)
        if self.LOGGING:
            model.compile(loss='mse',
                          optimizer=Adam(lr=self.alpha,
                                        clipvalue=1),
                          metrics=['mse'])
        else:
            model.compile(loss='mse',
                          optimizer=Adam(lr=self.alpha,
                                        clipvalue=1))
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
            TD_error = abs(predicted[0][action] - target)
            predicted[0][action] = target
            logs = self.model.train_on_batch(state, predicted)
            if self.LOGGING:
                self.tensorboard.on_epoch_end(self.iteration,
                                              self._named_logs(self.model, logs),
                                              TD_error=TD_error,
                                              reward=reward)
                self.iteration += 1

    # the DQN algorithm. some of the algorithm is moved to the replay method
    # TODO: preinitialize, then always add a 4-stack of frames to memory.
    # This will have consequences for replay() as well
    def learn(self, n_episodes=1000):
        # Run DQN.learn(), then in a separate terminal run
        # "tensorboard --logdir ./logs" and then open 
        # "localhost:6006" in your browser to open TensorBoard
        if self.LOGGING:
            self.tensorboard = Custom_TensorBoard(
                log_dir="./logs/{}".format(self.sim.get_name()),
                histogram_freq=0,
                batch_size=1,
                write_graph=True,
                write_grads=True
            )
            self.tensorboard.set_model(self.model)
            self.iteration = 0

        batch_size = self.METADATA['batch_size']

        target_update_counter = self.target_update_cnt
        for episode in range(n_episodes):
            t0 = time.time()
            total_reward = 0
            done = False
            # initialize state
            state = self.sim.reset()
            # model expects an array of samples, even if it is only one.
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
            if total_reward >= 0.8 * self.best_reward:
                self.best_reward = total_reward
                self.sim.render()

            print(f"Episode {episode + 1}: Total Reward --> {total_reward}")
            print(f"Epsilon: {self.eps}")
            print(f"Time taken: {time.time() - t0}\n")
            self.rewards.append(total_reward)
        if self.LOGGING:
            self.tensorboard.on_train_end(None)

    # play human first to collect valuable data for replay memory
    def run_human(self):
        import pickle
        from Misc import run_human
        self.load_memory()
        # collect data until memory is full
        status = "Running"
        while len(self.memory) < self.METADATA['memory_size'] and status != "Cancelled":
            status = run_human(self.sim, self)
            print("Memory Size: ", len(self.memory))
        # save collected data to file
        with open('human_data.dat', 'wb') as outfile:
            pickle.dump(self.memory, outfile)

    # load memory from pickle file
    def load_memory(self):
        import pickle
        with open('human_data.dat', 'rb') as infile:
            self.memory.extend(pickle.load(infile))
            print("Memory Size: ", len(self.memory))

    # show the predicted Q-Values for each action in state
    # TODO: think about how to best track these over time
    def show_best_action(self, state='Current'):
        key_map = {0:'N', 1:'S', 2:'E', 3:'W', 4:'D', 5:' '}
        if state == 'Current':
            state = self.sim.W.get_state()
            state = np.reshape(state, [1] + list(state.shape))

        QVals = self.model.predict(state)[0]
        maxval, maxidx = (-1000, -1)

        # TODO: Q-values over time, print at every run_human iteration
        print("| ", end="")
        for idx, val in enumerate(QVals):
            if val > maxval:
                (maxidx, maxval) = (idx, val)
            print(key_map[idx], ":", round(val, 2), " | ", end="")
        print(f" Best: {key_map[maxidx]}")

    # play the simulation by choosing optimal actions
    def play_optimal(self, eps=0):
        done = False
        state = self.sim.reset()
        while not done:
            self.sim.render()
            state = np.reshape(state, [1] + list(state.shape))
            self.show_best_action(state)
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
    def average_reward_per_k_episodes(self, k, plot=False):
        n_episodes = len(self.rewards)
        rewards_per_k = np.split(np.array(self.rewards), n_episodes/k)
        # TODO: fix this
        if plot:
            plt.plot(rewards_per_k)
            plt.title(f"Average reward per {k} episodes")
            plt.show()
        else:
            count = k
            print(f"Average reward per {k} episodes:")
            for r in rewards_per_k:
                print(count, ":", str(sum(r/k)))
                count += k

    # loads the weights of the model from file.
    # pass name to class initialization to use load
    def _load(self, name):
        self.model.load_weights(name)

    # saves the weights of the model to file
    def save(self, name):
        self.model.save_weights(name)

    # helper function to feed tensorboard the dict it wants
    def _named_logs(self, model, logs):
        result = {}
        for l in zip(model.metrics_names, logs):
            result[l[0]] = l[1]
        return result

