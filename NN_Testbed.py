from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# This is taken from the CartPole DQN example at the Keras GitHub page

class DQN_Example:
    def __init__(self, sim):
        self.sim = sim
        self.n_actions = sim.action_space.n
        self.model = False
        self.dqn = False

    def run_all(self, n_steps=50000, visualize=True, verbose=2, n_episodes=5):
        self.make_model()
        self.configure_and_compile()
        self.fit(n_steps, visualize, verbose)
        self.evaluate(n_episodes, visualize)

    def make_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(1,) + \
                        self.sim.observation_space.shape))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dense(self.n_actions))
        model.add(Activation('linear'))
        print(model.summary())
        self.model = model
        return model

    def configure_and_compile(self):
        memory = SequentialMemory(limit=50000, 
                                  window_length=1)
        policy = BoltzmannQPolicy()
        dqn = DQNAgent(model=self.model,
                        nb_actions=self.n_actions,
                        memory=memory,
                        nb_steps_warmup=10,
                        target_model_update=1e-2,
                        policy=policy)
        dqn.compile(Adam(lr=1e-3),
                    metrics=['mae'])
        self.dqn = dqn
        return dqn

    def fit(self, n_steps=50000, visualize=True, verbose=2):
        self.dqn.fit(self.sim, 
                     nb_steps=n_steps,
                     visualize=visualize,
                     verbose=verbose)

    def save_weights(self):
        sim_name = self.sim.spec.id
        self.dqn.save_weights(f'dqn_{sim_name}_weights.h5f', 
                              overwrite=True)

    def evaluate(self, n_episodes=5, visualize=True):
        self.dqn.test(self.sim, 
                      nb_episodes=n_episodes, 
                      visualize=visualize)
