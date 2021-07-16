import gym
from keras import optimizers
import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt
from collections import deque

from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Dense, Input
from tensorflow.python.keras.mixed_precision import policy,


class Agent:
    def __init__(self, action_space, observation_space):
        self.critic_step_size = 1e-1
        self.step_size = 1e-1
        self.epsilon = 1e-2

        self.discount = 0.8

        self.batch_size = 64
        
        self.action_space = action_space
        self.observation_space = observation_space

        self.discrete_obs_space_size = [20] * len(observation_space.high)
        self.discrete_os_win_size = (observation_space.high - observation_space.low) / self.discrete_obs_space_size

        
        self.actor_weights = np.zeros((self.discrete_obs_space_size + [action_space.n]))
        self.critic_weights = np.zeros(self.discrete_obs_space_size)
        self.avg_reward = 0

        self.memory = deque(maxlen=1e6)

        self.init_model()

    def init_model(self):
        self.q_eval = Sequential()
        self.q_eval.add( Input(len(self.observation_space.low)) )
        self.q_eval.add( Dense(64, activation='relu') )
        self.q_eval.add( Dense(32, activation='relu') )
        self.q_eval.add( Dense(self.action_space.n) )
        
        self.q_eval.compile(optimizer=Adam(lr=1e-2), loss='mse')

    def get_discrete_state(self, state:np.ndarray)->tuple:
        discrete_state = (state - self.observation_space.low) / self.discrete_os_win_size

        return tuple(discrete_state.astype(int))

    def softmax(self, actor_w:np.ndarray, state:np.ndarray, tau=1)->np.ndarray:
            action_values = actor_w[state]
            c = np.max(action_values)

            num = np.exp((action_values - c)/tau)
            den = np.sum(np.exp((action_values - c))/tau)
            return num/den

    def policy(self, state:tuple)->int:
        state = self.get_discrete_state(state)
        if np.random.random()>self.epsilon:
            action = np.argmax(self.q_eval.predict(state))
        else:
            action = np.random.choice(np.arange(0,self.action_space.n))
        return action

    def start(self, state):
        self.last_action = self.policy(state)
        self.last_state  = state

    def step(self, state:np.ndarray, reward:float, done:bool):
        self.memory.append((self.last_state, self.last_action, reward, state, done))
        action = policy(state)
        
        self.last_state = state
        self.last_action = action
        return action

    def sample_buffer(self, batch_size)->tuple:
        last_state, last_action, reward, state, done = [],[],[],[],[] 
        for i in range(batch_size):
            indx = np.random.randint(self.memory.maxlen)
            last_state.append(self.memory[indx][0])
            last_action.append(self.memory[indx][1])
            reward.append(self.memory[indx][2])
            state.append(self.memory[indx][3])
            done.append(self.memory[indx][4])
        return np.array(last_state), np.array(last_action), np.array(reward), np.array(state), np.array(done)

    def learn(self):
        if len(self.sample_buffer)>=self.sample_buffer.maxlen:
            
            last_state, last_action, reward, state, done = self.sample_buffer(self.batch_size)
            
            state_action_value = self.q_eval.predict(self.last_state)
            next_state_action_value = self.q_eval.predict(state)
            
            state_action_value_target = state_action_value.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            state_action_value_target[batch_index, action]


    