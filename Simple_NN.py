from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import random

# suppress unecessary warnings from tensorflow
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

# how much data to generate
data_size = 10

# make a single neuron
model = Sequential()
layer = Dense(units=1, activation='sigmoid', input_dim=1)
model.add(layer)
model.compile(loss='mse', optimizer=Adam())

# make some data (random nums in [0,1] + rounding result)
data_in  = [np.array([random.uniform(0, 1)]) for i in range(data_size)]
data_out = [1 if d > 0.5 else 0 for d in data_in]

# predict something
print(f"Before: {data_in[0]} --> {model.predict(x=data_in[0])}")

# fit the model
print("Fitting...")
model.fit(x=np.array(data_in), y=np.array(data_out), batch_size=None, epochs=10)

# predict something again
print(f"After: {data_in[0]} --> {model.predict(x=data_in[0])}")

