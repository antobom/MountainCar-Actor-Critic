import gym
from keras import optimizers
import numpy as np
from collections import deque
import time
# import matplotlib.pyplot as plt
from collections import deque

from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Dense, Input

class ReplayBuffer:
    def __init__(self, max_size, state_size, action_shape):
        self.max_size = max_size
        
        self.previous_state_memory = np.zeros((self.max_size, state_size))
        self.action_memory = np.zeros((self.max_size, action_shape), dtype=np.int8)
        self.reward_memory = np.zeros((self.max_size))
        self.state_memory = np.zeros((self.max_size, state_size))
        self.terminal_memory = np.zeros((self.max_size), dtype=np.bool)
        self.memory_cntr = 0

    def store_transition(self, previous_state:np.ndarray, last_action:np.ndarray, reward:float, state:np.ndarray, done:bool):
        indx = self.memory_cntr%self.max_size
        self.previous_state_memory[indx] = previous_state
        self.action_memory[indx] = last_action
        self.reward_memory[indx] = reward
        self.terminal_memory[indx] = done
        
        self.memory_cntr+=1

    def sample_buffer(self, batch_size):
        batch_size = min(batch_size, self.max_size)
        # current_size = min(self.memory_cntr, self.max_size)

        batch = np.random.choice(self.max_size, batch_size)

        previous_states = self.previous_state_memory[batch]
        last_actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states = self.state_memory[batch]
        terminal = self.terminal_memory[batch]

        return previous_states, last_actions, rewards, states, terminal

class DeepQAgent:
    def __init__(self, name, action_space, observation_space, epsilon, epsilon_decay, epsilon_end, discount, batch_size, learning_rate, memory_size=1000_000):
        
        self.name = name
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end

        self.discount = discount

        self.batch_size = batch_size
        
        self.action_space = action_space
        self.observation_space = observation_space
        self.memory_size = memory_size

        self.memory = ReplayBuffer(self.memory_size, len(self.observation_space.low), 1)

        self.init_model(learning_rate)

    def init_model(self, lr):
        self.q_eval = Sequential()
        self.q_eval.add( Input( (len(self.observation_space.low), ) ) )
        self.q_eval.add( Dense(256,activation='relu') )
        self.q_eval.add( Dense(128, activation='relu') )
        self.q_eval.add( Dense(self.action_space.n) )
        
        self.q_eval.compile(optimizer=Adam(lr=lr), loss='mse')
        print(self.q_eval.summary())

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
        # state = self.get_discrete_state(state)
        if np.random.random()>self.epsilon:
            action = np.argmax(self.q_eval.predict(np.array([state])))
        else:
            action = np.random.choice(np.arange(0,self.action_space.n))
        return action

    def start(self, state) -> int:
        self.last_action = self.policy(state)
        self.last_state  = state
        return self.last_action

    def step(self, state:np.ndarray, reward:float, done:bool):
        self.memory.store_transition(self.last_state, self.last_action, reward, state, done)
        action = self.policy(state)
        
        self.last_state = state
        self.last_action = action
        return action

    def learn(self):
        if self.memory.memory_cntr>self.batch_size:
            last_states, last_actions, reward, state, done = self.memory.sample_buffer(self.batch_size)

            action_values = self.q_eval.predict(last_states)
            next_action_values = self.q_eval.predict(state)

            target = action_values.copy()
            state_index = np.arange(self.batch_size, dtype=np.int8)

            # target[state_index, last_actions] = reward + ( self.discount*np.max(next_action_values, axis=1) - action_values[state_index, last_actions] ) * (1-done)
            target[state_index, last_actions] = reward + self.discount*np.max(next_action_values, axis=1) * (1-done)
            
            self.q_eval.fit(last_states, target, verbose=0)

            self.epsilon = self.epsilon*self.epsilon_decay if self.epsilon>self.epsilon_end else self.epsilon_end

    def save_model(self, path:str):
        self.q_eval.save(path)


def training_loop(agent:DeepQAgent, env:gym.Env, episodes:int, show_every=1):
    train_reward = deque(maxlen=600)
    for episode in range(1, episodes):
        done = False
        episode_reward = 0
        action = agent.start(env.reset())
        while not done:
            
            new_state, reward, done, _ = env.step(action)
            episode_reward+=reward
            action = agent.step(new_state, reward, done)

            agent.learn()
            env.render()

        train_reward.append(episode_reward)
        if episode%show_every==0:
            mean_reward = np.mean(train_reward)
            print("agent: {} | episode: {:3d} | episode_reward: {:5d} | mean reward: {:4.4f} | epsilon: {:3.3f}".format(agent.name, episode, int(episode_reward), mean_reward, agent.epsilon))
            if np.mean(episode_reward>-150) and episode%50:
                agent.save_model(f"models/model_{agent.name}_{episode}.h5")
    agent.save_model(f"models/model_{agent.name}_{0}.h5")
    

    return agent, train_reward


if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    agent = DeepQAgent(name = "agent2", action_space=env.action_space, observation_space=env.observation_space, \
    epsilon=1, epsilon_decay=0.996, epsilon_end=1e-2, discount=0.99, batch_size=1024, learning_rate=5e-3)

    training_loop(agent, env, episodes=500, show_every=1)
    

