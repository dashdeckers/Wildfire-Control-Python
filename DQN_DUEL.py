from DQN import DQN
from collections import deque

import numpy as np
import time, random

# Dueling DQN specific imports
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Activation
from keras.optimizers import Adam
from keras import backend as K

class DQN_DUEL(DQN):
    def __init__(self, sim, name="no_name", verbose=True):
        DQN.__init__(self, sim, name, verbose)

    def make_network(self):
        input_shape = (self.sim.W.WIDTH, self.sim.W.HEIGHT, self.sim.W.DEPTH)
        
        # Create input layer according to input_shape and flatten it
        input_layer = Input(shape = input_shape)
        flatten = Flatten()(input_layer)

        # Advantage stream connected to flatten, output size is action_size
        dense1 = Dense(units=50, activation='sigmoid')(flatten)
        advantage = Dense(self.action_size, activation='linear')(dense1)

        # Value stream connected to flatten, output size is 1
        dense2 = Dense(units=50, activation='sigmoid')(flatten)
        value = Dense(1, activation='linear')(dense2)

        # Combine both streams using: q = a - mean(a) + v
        output_layer = Lambda(
                lambda x: x[0]-K.mean(x[0])+x[1],
                output_shape = (self.action_size,)
            )([advantage, value])
        output_layer = Activation('linear')(output_layer)

        # Create model by setting input and output
        model = Model(inputs=[input_layer], outputs=[output_layer])
        # Copied from DQN, need to understand more
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha, clipvalue=1))

        # Print summary and return resulting model
        if self.verbose:
            model.summary()
        return model
