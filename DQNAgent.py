from __future__ import absolute_import
from __future__ import print_function
from sumolib import checkBinary

import os
import sys
import optparse
import subprocess
import random
import traci
import random
import numpy as np
import keras
import h5py
from collections import deque
from keras.layers import Input, Conv2D, Flatten, Dense
from keras.models import Model

class DQNAgent:
    def __init__(self):
        self.gamma = 0.95   # gamma  0,95
        self.epsilon = 1  # raziskovanje   v main definiramo decay
        self.learning_rate = 0.0002   #0,0002
        self.allActions = 2  # stevilo moznih akcij

        self.memory = deque(maxlen=200)
        self.model = self._build_model()


    def _build_model(self):
		# build cnn
        input_1 = Input(shape=(12, 12, 1))
        x1 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input_1)
        x1 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(x1)
        x1 = Flatten()(x1)

        input_2 = Input(shape=(12, 12, 1))
        x2 = Conv2D(16, (4, 4), strides=(2, 2), activation='relu')(input_2)
        x2 = Conv2D(32, (2, 2), strides=(1, 1), activation='relu')(x2)
        x2 = Flatten()(x2)

        input_3 = Input(shape=(2, 1))
        x3 = Flatten()(input_3)

        x = keras.layers.concatenate([x1, x2, x3])
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(2, activation='linear')(x)

        model = Model(inputs=[input_1, input_2, input_3], outputs=[x])
        model.compile(optimizer=keras.optimizers.RMSprop(
            lr=self.learning_rate), loss='mse')

        return model

    def save(self, name):
        self.model.save_weights(name)


    #returns action
    #0 = vertical open
    #1 = horizontal open
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            a = random.randrange(self.allActions)
            return random.randrange(self.allActions)
        else:
            tmp = self.model.predict(state)
            a = np.argmax(tmp[0])
        return a

    def lrn(self, batch_size):
        tempbatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in tempbatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def load(self, name):
        self.model.load_weights(name)
