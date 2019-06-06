from DQN import DQN
from collections import deque

import numpy as np
import time, random

class DQN_SARSA(DQN):
    def __init__(self, sim, name="no_name", verbose=True):
        DQN.__init__(self, sim, name, verbose)
    
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

            # Initialize the action
            action = self.choose_action(state)

            # Start the simulation episode
            while not done:
                # Execute an action following the e-greedy policy
                sprime, reward, done, _ = self.sim.step(action)
                sprime = np.reshape(sprime, [1] + list(sprime.shape))

                aprime = self.choose_action(sprime)

                # Store the observed experience in memory
                self.remember(state, action, reward, sprime, aprime, done)

                # If we have collected enough experiences, learn from memory
                if len(self.memory) > self.METADATA['batch_size']:
                    self.replay()

                # Every set number of iterations, update the target network
                target_update_counter -= 1
                if target_update_counter == 0:
                    target_update_counter = self.target_update_freq
                    self.target.set_weights(self.model.get_weights())

                # Set the state and action to be the next state S' and action A'
                state = sprime
                action = aprime

                # Keep track of the rewards and the total accumulated reward
                total_reward += reward
                rewards.append(reward)

            # Keep track of agent deaths
            if self.DEBUG > 0:
                if len(self.sim.W.agents) == 0:
                    self.logs['agent_deaths'].append(True)
                else:
                    self.logs['agent_deaths'].append(False)

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

        for state, action, reward, sprime, aprime, done in batch:
            # Get the prediction for the state S
            prediction = self.target.predict(state)[0]

            # If S was a terminal state, the cumulative reward from that
            # state forward is simply the reward recieved for that state
            if done:
                prediction[action] = reward
            # Otherwise, estimate the cumulative reward from that state
            # forward by using the Q-value of the state S' and the action
            # A' as a proxy (=bootstrapping via TD methods)
            else:
                predQ = self.target.predict(sprime)[0][aprime]
                prediction[action] = reward + self.gamma * predQ

            # Store the states and their updated predictions for this batch
            states_batch.append(state[0])
            predicted_batch.append(prediction)

        # Convert the batch into numpy arrays for Keras and fit the model
        states = np.array(states_batch)
        predictions = np.array(predicted_batch)
        self.model.fit(states, predictions, epochs=1, verbose=0)

    # Store an experience in memory
    def remember(self, state, action, reward, sprime, aprime, done):
        self.memory.append((state, action, reward, sprime, aprime, done))

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
            # Initialize action
            action = self.choose_randomwalk_action()

            while not done:
                # Observe sprime and reward
                sprime, reward, done, _ = self.sim.step(action)
                sprime = np.reshape(sprime, [1] + list(sprime.shape))

                # Get aprime
                aprime = self.choose_randomwalk_action()

                # Collect memories
                memories.append((state, action, reward, sprime, aprime, done))

                # Set new action and state for the next iteration
                state = sprime
                action = aprime

                # Only if we contained the fire, we collect the memories
                if reward == self.METADATA['contained_bonus']:
                    success_count += 1
                    # Store successful experience in memory
                    for state, action, reward, sprime, aprime, done in memories:
                        self.remember(state, action, reward, sprime, aprime, done)
                    # Done is true, but not if it were a real run, so set this after
                    # storing the transitions in memory
                    done = True
                    # Collect logging info and return
                    if success_count == num_of_successes:
                        self.logs['init_memories'] = len(self.memory)
                        return
