from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, InputLayer, Flatten
from keras.optimizers import Adam

from tensorflow.python.keras import activations
from collections import deque
import time
import numpy as np



REPLAY_MEMORY_SIZE = 50_000

class DQNAgent:
    def __init__(self):
        # main model get tained every step 
        self.model = self.create_model()
        # Target model is updated after a certain amout of steps, it is the one who predicts 
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())


        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0


    def create_model(self):
        model = Sequential()
        model.add(InputLayer(env.OBSERVATION_SPACE_VALUE))

        model.add(Flatten())
        model.add((Dense(64)))

        model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition:tuple):
        self.replay_memory.append(transition)

    def get_qs(self, state, step):
        return self.model_predict(np.array(state).reshape(-1, state.shape/255))[0]

    def get_discrete_state(state:np.ndarray):
        discrete_state:np.ndarray = (state - env.observation_space.low) / discrete_os_win_size
        return tuple(discrete_state.astype(int))