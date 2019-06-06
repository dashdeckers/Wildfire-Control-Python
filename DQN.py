from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

import numpy as np
import random, time, keras, json, os
from collections import deque

class DQN:
    def __init__(self, sim, name="no_name"):
        # Constants and such
        self.sim = sim
        self.name = name
        self.METADATA = sim.METADATA
        self.action_size = self.sim.n_actions
        self.DEBUG = sim.DEBUG

        # DQN memory
        self.memory = deque(maxlen=self.METADATA['memory_size'])

        # Information to save to file
        self.logs = {
            'best_reward'   : -10000,
            'total_rewards' : list(),
            'agent_pos'     : list(),
            'maps'          : list(),
            'consec_wins'   : 0,
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

        # Print Constants
        print("\n\t[Parameters]")
        print("[decay]", self.METADATA['eps_decay_rate'])
        print("[alpha]", self.METADATA['alpha'])
        print("[gamma]", self.METADATA['gamma'])
        print("[batch]", self.METADATA['batch_size'])
        print("[wind speed]", self.METADATA['wind'][0])
        print("[target upd]", self.METADATA['target_update'])

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
                # Save the state of the map
                if self.DEBUG > 0:
                    self.logs['maps'].append([episode, map_string])

            # Print some information about the episode
            print(f"[Episode {episode + 1}]\tTime: {round(time.time() - t0, 3)}")
            print(f"\t\tEpsilon: {round(self.eps, 3)}")
            print(f"\t\tAgent dead: {len(self.sim.W.agents) == 0}")
            print(f"\t\tReward: {total_reward}\n")

            # Decay the epsilon value for the next episode
            self.decay_epsilon(episode)

            # Log the rewards over time
            self.logs['total_rewards'].append(total_reward)

            # Stop condition, if the agent solves the environment X times in a row
            if self.METADATA['stop_early']:
                if total_reward > 1500:
                    self.logs['consec_wins'] += 1
                else:
                    self.logs['consec_wins'] = 0
                if self.logs['consec_wins'] >= self.METADATA['stop_early']:
                    self.logs['n_episodes'] = episode
                    print("Stopping early!")
                    break

        # Save the total time taken for this run
        self.logs['total_time'] = round(time.time() - start_time, 3)

        # Write logs and model to file
        self.write_data()

    # Fit the model with a random sample taken from the memory
    def replay(self):
        states_batch = list()
        predicted_batch = list()

        # Sample a random batch of memories from memory
        batch = random.sample(self.memory, self.METADATA['batch_size'])

        for state, action, reward, sprime, done in batch:
            # Get the prediction for the state S
            prediction = self.target.predict(state)[0]

            # If S was a terminal state, the cumulative reward from that
            # state forward is simply the reward recieved for that state
            if done:
                prediction[action] = reward
            # Otherwise, estimate the cumulative reward from that state
            # forward by using the maximum Q-value of the state S' as a
            # proxy (=bootstrapping via TD methods)
            else:
                predQ = np.amax(self.target.predict(sprime)[0])
                prediction[action] = reward + self.gamma * predQ

            # Store the states and their updated predictions for this batch
            states_batch.append(state[0])
            predicted_batch.append(prediction)

        # Convert the batch into numpy arrays for Keras and fit the model
        states = np.array(states_batch)
        predictions = np.array(predicted_batch)
        self.model.fit(states, predictions, epochs=1, verbose=0)

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
        while not done:
            self.sim.render()
            state = np.reshape(state, [1] + list(state.shape))
            self.show_info(state)
            action = self.choose_action(state, eps=eps)
            state, reward, done, _ = self.sim.step(action)
            total_reward += reward
            time.sleep(0.1)
        self.sim.render()
        print(f"Total reward: {total_reward}")

    # Show the Q-values for each action in the state, the best action, and wind info
    def show_info(self, state):
        w_speed, w_vector = self.sim.W.wind_speed, self.sim.W.wind_vector
        print(f"Wind Speed: {w_speed}")
        print(f"Wind direction: {w_vector}")

        # Predict the Q-values via the network
        QVals = self.model.predict(state)[0]

        # Print the Q-values and their maximum
        key_map = {0:'N', 1:'S', 2:'E', 3:'W', 4:'D', 5:' '}
        print("| ", end="")
        for idx, val in enumerate(QVals):
            val = round(val, 2)
            direction = key_map[idx]
            extra_space = " " if val > 0 else ""
            print(f"{direction} : {extra_space}{val:.2f} | ", end="")
            if idx == 1:
                print("\n| ", end="")
        print(f"\nBest Action: {key_map[np.argmax(QVals)]}\n")

    '''
    Collects memories as follows:

    Make the agent walk somewhat randomly but always such that it walks clockwise around
    fire. So if the agent is below and to the right of the fire it will chose an action
    to either go left or downwards until it is below the fire and then it will go either
    up or left. When the fire is contained, the simulation is reset.
    
    It only collects the memories that lead to a successful containment of the fire.
    '''
    def collect_memories(self, num_of_successes=100):
        if not num_of_successes:
            return
        # Wipe internal memory
        self.memory = deque()
        success_count = 0
        # While memory is not filled up
        while True:
            memories = []
            done = False
            # Initialize state
            state = self.sim.reset()
            state = np.reshape(state, [1] + list(state.shape))

            while not done:
                # Choose an action
                action = self.choose_randomwalk_action()

                # Observe sprime and reward
                sprime, reward, done, _ = self.sim.step(action)
                sprime = np.reshape(sprime, [1] + list(sprime.shape))

                # Collect memories
                memories.append((state, action, reward, sprime, done))
                state = sprime

                # Only if we contained the fire, we collect the memories
                if reward == self.METADATA['contained_bonus']:
                    success_count += 1
                    # Store successful experience in memory
                    for state, action, reward, sprime, done in memories:
                        self.remember(state, action, reward, sprime, done)
                    # Done is true, but not if it were a real run, so set this after
                    # storing the transitions in memory
                    done = True
                    # Collect logging info and return
                    if success_count == num_of_successes:
                        self.logs['init_memories'] = len(self.memory)
                        return

    # Choose an action depending on the agents position relative to the fire.
    # The action should be safe, if possible
    def choose_randomwalk_action(self):
        key_map = {'N':0, 'S':1, 'E':2, 'W':3}
        width, height = self.sim.W.WIDTH, self.sim.W.HEIGHT
        agent_x, agent_y = self.sim.W.agents[0].x, self.sim.W.agents[0].y
        mid_x, mid_y = (int(width / 2), int(height / 2))

        # Loop to try to avoid choosing actions that lead to death
        count = 0
        while True:
            # The chosen action should always make the agent go around the fire
            if agent_x >= mid_x and agent_y > mid_y:
                possible_actions = ["S", "W"]
            if agent_x > mid_x and agent_y <= mid_y:
                possible_actions = ["S", "E"]
            if agent_x <= mid_x and agent_y < mid_y:
                possible_actions = ["N", "E"]
            if agent_x < mid_x and agent_y >= mid_y:
                possible_actions = ["N", "W"]

            # Choose randomly from valid actions
            action = key_map[np.random.choice(possible_actions)]

            # Break when it is a safe move or when we have tried too often
            fire_at_loc = self.sim.W.agents[0].fire_in_direction(action)
            if not fire_at_loc or count > 10:
                break
            count += 1

        return action

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
        name = self.sim.get_name(int(n_episodes), memories, self.name)
        counter = 0
        while os.path.isfile("Logs/" + name) or os.path.isfile("Models/" + name):
            if counter > 0:
                n_digits_to_delete = len(str(counter))
                name = name[:-n_digits_to_delete]
            name = name + str(counter)
            counter += 1

        # If the folders don't exist, create them
        if not os.path.exists("Logs/"):
            os.makedirs("Logs/")
        if not os.path.exists("Models/"):
            os.makedirs("Models/")

        # Write model
        self.save_model(name)

        # Write logs
        with open("Logs/" + name, 'w') as file:
            json.dump(self.logs, file)

    # Loads the weights of the model from file.
    def load_model(self):
        all_names = os.listdir("Models")
        print("\nChoose a model from the list below to load:")
        for idx, name in enumerate(all_names):
            print(f"\t[{idx}] {name}")
        try:
            selection = int(input(f"Select one [0-{idx}]: \n"))
            name = os.path.join("Models", all_names[selection])
            self.model.load_weights(name)
            print("Model loaded!")
        except ValueError as e:
            print("Invalid Selection")

    # Saves the weights of the model to file
    def save_model(self, name):
        name = "Models/" + name
        self.model.save_weights(name)
