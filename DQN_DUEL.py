from DQN import DQN
from collections import deque

import numpy as np
import time, random

# Dueling DQN specific imports
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Lambda
from keras.optimizers import Adam
from keras import backend as K

class DQN_DUEL(DQN):
    def __init__(self, sim, name="no_name", verbose=True):
        DQN.__init__(self, sim, name, verbose)

### DUELING DQN NETWORK
    def make_network(self):
        model = Sequential()
        input_shape = (self.sim.W.WIDTH, self.sim.W.HEIGHT, self.sim.W.DEPTH)
        
        # Create input layer according to input_shape and flatten it
        input_layer = Input(shape = input_shape)
        flatten = Flatten()(input_layer)

        # Advantage stream connected to flatten, output size is action_size
        dense1 = Dense(units=25, activation='sigmoid')(flatten)
        advantage = Dense(self.action_size, activation='linear')(dense1)

        # Value stream connected to flatten, output size is 1
        dense2 = Dense(units=25, activation='sigmoid')(flatten)
        value = Dense(units=1, activation='linear')(dense2)

        # Combine advantage & value streams into output layer
        #		Formula: q = v + (a - mean(a))
        def merger(streams):
            adv, val = streams
            return val + (adv - K.mean(adv, axis=1, keepdims=True))
        output_layer = Lambda(merger)([advantage, value])

        # Create model by defining input and output layers
        model = Model(inputs=[input_layer], outputs=[output_layer])
        # Set the loss function and optimizer
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha, clipvalue=1))

        # Print summary and return resulting model
        if self.verbose:
            model.summary()
        return model