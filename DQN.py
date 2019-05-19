from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

import numpy as np
import random, time, keras, json, os
from collections import deque

class DQN:
    def __init__(self, sim):
        # Constants and such
        self.sim = sim
        self.METADATA = sim.METADATA
        self.action_size = self.sim.n_actions
        self.DEBUG = sim.DEBUG

        # DQN memory
        self.memory = deque(maxlen=self.METADATA['memory_size'])

        # Information to save to file
        self.logs = {
            'best_reward'   : -10000,
            'total_rewards' : list(),
            'all_rewards'   : list(),
            'TD_errors'     : list(),
            'wind_values'   : list(),
            'agent_pos'     : list(),
            'epsilons'      : list(),
            'maps'          : list(),
            'deaths'        : 0,
            'init_memories' : 0,
            'total_time'    : 0,
            'n_episodes'    : 0,
        }

        # DQN Parameters
        self.max_eps = self.METADATA['max_eps']
        self.min_eps = self.METADATA['min_eps']
        self.eps_decay_rate = self.METADATA['eps_decay_rate']
        self.eps = self.max_eps
        self.gamma = self.METADATA['gamma']
        self.alpha = self.METADATA['alpha']
        self.target_update_freq = self.METADATA['target_update']

        # Network and target network
        self.model = self.make_network()
        self.target = keras.models.clone_model(self.model)
        self.target.set_weights(self.model.get_weights())

    '''
    Main methods related to learning
    '''

    # The learning algorithm
    def learn(self, n_episodes=1000):
        # Time the entire run
        start_time = time.time()

        # Save the total number of episodes
        self.logs['n_episodes'] = n_episodes

        # Initialize counter to update the target network in intervals        
        target_update_counter = self.target_update_freq

        # Start the main learning loop
        for episode in range(n_episodes):

            # Every 100 epsiodes, see how the agent is behaving
            if self.DEBUG > 1 and episode % 100 == 0 and not episode == 0:
                self.play_optimal(self.eps)

            # Initialze the done flag, the reward accumulator, time, rewards etc
            done = False
            total_reward = 0
            t0 = time.time()
            rewards = list()

            # Initialize the state, and reshape because Keras expects the
            # first dimension to be the batch size
            state = self.sim.reset()
            state = np.reshape(state, [1] + list(state.shape))

            # Keep track of the agent starting positions
            if self.DEBUG > 0:
                agent_x, agent_y = self.sim.W.agents[0].x, self.sim.W.agents[0].y
                self.logs['agent_pos'].append((agent_x, agent_y))

            # Start the simulation episode
            while not done:
                # Execute an action following the e-greedy policy
                action = self.choose_action(state)
                sprime, reward, done, _ = self.sim.step(action)
                sprime = np.reshape(sprime, [1] + list(sprime.shape))

                # Store the observed experience in memory
                self.remember(state, action, reward, sprime, done)

                # If we have collected enough experiences, learn from memory
                if len(self.memory) > self.METADATA['batch_size']:
                    self.replay()

                # Every set number of iterations, update the target network
                target_update_counter -= 1
                if target_update_counter == 0:
                    target_update_counter = self.target_update_freq
                    self.target.set_weights(self.model.get_weights())

                # Set the state S to be the next state S', for the next iteration
                state = sprime

                # Keep track of the rewards and the total accumulated reward
                total_reward += reward
                rewards.append(reward)

            # If the last episode was somewhat successful, render its final state
            if total_reward >= 0.9 * self.logs['best_reward'] or total_reward > 300:
                map_string = self.sim.render()
                if total_reward > self.logs['best_reward']:
                    self.logs['best_reward'] = total_reward
                # Save the final state of the map
                if self.DEBUG > 0:
                    self.logs['maps'].append([episode, map_string])

            # Keep track of how often the agent is dying
            if self.DEBUG > 0 and len(self.sim.W.agents) == 0:
                self.logs['deaths'] += 1

            # Keep track of the wind speeds and directions
            if self.DEBUG > 0:
                speed, direction = self.sim.W.wind_speed, self.sim.W.wind_vector
                self.logs['wind_values'].append((speed, direction))

            # Print some information about the episode
            print(f"[Episode {episode + 1}]\tTime: {round(time.time() - t0, 3)}")
            print(f"\t\tEpsilon: {round(self.eps, 3)}")
            print(f"\t\tAgent dead: {len(self.sim.W.agents) == 0}")
            print(f"\t\tReward: {total_reward}\n")

            # Log and decay the epsilon value for the next episode
            if self.DEBUG > 1: self.logs['epsilons'].append(self.eps)
            self.decay_epsilon(episode)

            # Collect data on the rewards over time
            if self.DEBUG > 0: self.logs['total_rewards'].append(total_reward)
            if self.DEBUG > 1: self.logs['all_rewards'].append(rewards)

        # Save the total time taken for this run
        self.logs['total_time'] = round(time.time() - start_time, 3)

        # Write logs and model to file
        self.write_data()

    # Fit the model with a random sample taken from the memory
    def replay(self):
        states_batch = list()
        predicted_batch = list()
        td_errors = list()

        # Sample a random batch of memories from memory
        batch = random.sample(self.memory, self.METADATA['batch_size'])

        for state, action, reward, sprime, done in batch:
            # Get the prediction for the state S
            prediction = self.target.predict(state)[0]

            # Save the current Q-value estimate to calculate TD error
            old_predicted_qval = prediction[action]

            # If S was a terminal state, the cumulative reward from that
            # state forward is simply the reward recieved for that state
            if done:
                prediction[action] = reward
            # Otherwise, estimate the cumulative reward from that state
            # forward by using the maximum Q-value of the state S' as a
            # proxy (=bootstrapping)
            else:
                maxQ = np.amax(self.target.predict(sprime)[0])
                prediction[action] = reward + self.gamma * maxQ

            # The TD error is the difference between the old predicted
            # Q-value for the state S and action A and the new prediction
            td_errors.append(abs(float(old_predicted_qval - prediction[action])))

            # Store the states and their updated predictions for this batch
            states_batch.append(state[0])
            predicted_batch.append(prediction)

        # Convert the batch into numpy arrays for Keras and fit the model
        states = np.array(states_batch)
        predictions = np.array(predicted_batch)
        self.model.fit(states, predictions, epochs=1, verbose=0)

        # Collect the data on TD errors
        if self.DEBUG > 1: self.logs['TD_errors'].append(td_errors)

    # Choose an action A based on state S following the e-greedy policy
    def choose_action(self, state, eps=None):
        # Epsilon is either taken from current value, or passed as argument
        eps_threshold = self.eps if eps is None else eps

        # Either choose action with highest Q-value or a random action
        if random.uniform(0, 1) > eps_threshold:
            return np.argmax(self.model.predict(state)[0])
        else:
            return np.random.choice(self.METADATA['n_actions'])

    # Decay the epsilon value in a rate proportional to the episode number
    def decay_epsilon(self, episode_num=None):
        self.eps = self.min_eps \
                    + (self.max_eps - self.min_eps) \
                    * np.exp(-self.eps_decay_rate * episode_num)

    # Store an experience in memory
    def remember(self, state, action, reward, sprime, done):
        self.memory.append((state, action, reward, sprime, done))

    # Create the neural network
    def make_network(self):
        input_shape = (self.sim.W.WIDTH, self.sim.W.HEIGHT, self.sim.W.DEPTH)
        layers = [
            Flatten(input_shape=input_shape),
            # One hidden layer with 50 neurons and a sigmoid activation function
            Dense(units=50,
                  activation='sigmoid'),
            # Output layer has a linear activation function
            Dense(units=self.action_size,
                  activation='linear'),
        ]
        '''
        TODO: Consider using an initializer for the layers:
        bias_initializer='random_uniform',
        kernel_initializer='random_uniform'),
        '''
        model = Sequential(layers)
        # Compile model with mean squared error loss metric
        model.compile(loss='mse',
                      # And an Adam optimizer with gradient clipping
                      optimizer=Adam(lr=self.alpha,
                                     clipvalue=1))
        model.summary()
        return model

    '''
    Miscellaneous, helper methods:
    '''

    # Play the simulation by following the optimal policy
    def play_optimal(self, eps=0):
        done = False
        total_reward = 0
        state = self.sim.reset()
        w_speed, w_vector = self.sim.W.wind_speed, self.sim.W.wind_vector
        while not done:
            self.sim.render()
            state = np.reshape(state, [1] + list(state.shape))
            print(f"Wind Speed: {w_speed}, Wind direction: {w_vector}")
            self.show_best_action(state)
            action = self.choose_action(state, eps=eps)
            state, reward, done, _ = self.sim.step(action)
            total_reward += reward
            time.sleep(0.1)
        self.sim.render()
        print(f"Total reward: {total_reward}")

    # Show the Q-values for each action in the current state, and show the highest one
    def show_best_action(self, state='Current'):
        if state == 'Current':
            state = self.sim.W.get_state()
            state = np.reshape(state, [1] + list(state.shape))

        # Predict the Q-values via the network
        QVals = self.model.predict(state)[0]

        # Print the Q-values and their maximum
        key_map = {0:'N', 1:'S', 2:'E', 3:'W', 4:'D', 5:' '}
        print("| ", end="")
        for idx, val in enumerate(QVals):
            print(key_map[idx], ":", round(val, 2), " | ", end="")
        print(f" Best: {key_map[np.argmax(QVals)]}")

    '''
    Automatically fills memories as follows:

    Create a map with a dirt/road circle around the fire of varying radius via the
    mid point circle drawing algorithm and store that state along with the large
    containment reward that comes with it in memory
    '''
    def collect_memories(self, num_of_memories=1000):
        # Wipe internal memory
        self.memory = deque()
        # Get the possible values for circle radius
        width, height = self.sim.W.WIDTH, self.sim.W.HEIGHT
        smallest_dimension = width if width > height else height
        possible_radiuses = np.array(range(1, int(smallest_dimension / 2)))
        midpoint = (int(width / 2), int(height / 2))
        # Collect the memories
        for i in range(num_of_memories):
            # Generate a random circle
            circle = [np.random.choice(possible_radiuses), midpoint]
            # Get the state S from a the circle
            state = self.sim.reset(circle)
            state = np.reshape(state, [1] + list(state.shape))
            # Get the reward and next state S' from closing the circle
            sprime, reward, done, _ = self.sim.step("D")
            sprime = np.reshape(sprime, [1] + list(sprime.shape))
            # Make sure we actually got the containment bonus
            assert reward == 1000
            # We pretend a movement action resulted in this reward
            action = np.random.choice(np.array([0, 1, 2, 3]))
            # Store experience in memory
            self.remember(state, action, reward, sprime, done)
        # Collect logging info
        if self.DEBUG > 0: self.logs['init_memories'] = len(self.memory)

    # Writes the logs and the metadata to a file with an appropriate name
    def write_data(self):
        # Also save metadata
        self.logs['metadata'] = self.METADATA

        # Create filename
        n_episodes = self.logs['n_episodes']
        if n_episodes >= 1000:
            n_episodes /= 1000
        else:
            n_episodes = 0
        memories = self.logs['init_memories']
        size = self.sim.W.WIDTH
        name = self.sim.get_name(int(n_episodes), memories, size)
        counter = 0
        while os.path.isfile("Logs/" + name) or os.path.isfile("Models/" + name):
            if counter > 0:
                n_digits_to_delete = len(str(counter))
                name = name[:-n_digits_to_delete]
            name = name + str(counter)
            counter += 1

        # Write model
        self.save_model(name)

        # Write logs
        with open("Logs/" + name, 'w') as file:
            json.dump(self.logs, file)

    # Loads the weights of the model from file.
    def load_model(self, name):
        name = "Models/" + name
        self.model.load_weights(name)

    # Saves the weights of the model to file
    def save_model(self, name):
        name = "Models/" + name
        self.model.save_weights(name)
