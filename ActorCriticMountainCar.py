import gym
import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt
#from utils import display_mean_values




class Agent:
    def __init__(self, action_space, observation_space):
        self.critic_step_size = 1e-1
        self.actor_step_size = 1e-1
        self.avg_reward_step_size = 1e-6
        self.discount = 0.8
        
        self.action_space = action_space
        self.observation_space = observation_space

        self.discrete_obs_space_size = [20] * len(observation_space.high)
        self.discrete_os_win_size = (observation_space.high - observation_space.low) / self.discrete_obs_space_size

        
        self.actor_weights = np.zeros((self.discrete_obs_space_size + [action_space.n]))
        self.critic_weights = np.zeros(self.discrete_obs_space_size)
        self.avg_reward = 0



    def get_discrete_state(self, state:np.ndarray)->tuple:
        discrete_state = (state - self.observation_space.low) / self.discrete_os_win_size

        return tuple(discrete_state.astype(int))

    def softmax(self, actor_w:np.ndarray, state:int, tau=1)->np.ndarray:
        action_values = actor_w[state]
        c = np.max(action_values)

        num = np.exp((action_values - c)/tau)
        den = np.sum(np.exp((action_values - c))/tau)
        return num/den

    def policy(self, state:tuple)->int:
        self.softmax_prob = self.softmax(self.actor_weights, state)
        return np.random.choice(np.arange(0, self.action_space.n), p=self.softmax_prob)

    def start(self, state)->int:
        self.last_action = self.policy(state)
        self.last_state = state
        return self.last_action

    def step(self, state:tuple, reward:float)->int:
        delta           = reward  - self.avg_reward + self.critic_weights[state] - self.critic_weights[self.last_state]
        self.avg_reward = self.avg_reward + self.avg_reward_step_size * delta

        self.critic_weights[self.last_state] = self.critic_weights[self.last_state] + self.critic_step_size * delta 
        for a in range(self.action_space.n):
            if self.last_action==a:
                self.actor_weights[self.last_state][a] = self.actor_weights[self.last_state][a] + self.actor_step_size * delta * (1 - self.softmax_prob[a])
            else:
                self.actor_weights[self.last_state][a] = self.actor_weights[self.last_state][a] + self.actor_step_size * delta * (0 - self.softmax_prob[a])
        action = self.policy(state)

        self.last_action = action
        self.last_state = state
        
        return action

    
    def get_state_int(self, state:tuple)->np.ndarray:
        i = None
        return state




def training_loop(agent:Agent, env:gym.Env, episodes:int, show_every:int=1000):
    train_reward = deque(maxlen=500)
    for episode in range(1, episodes):

        done=False
        episode_reward = 0
        state = agent.get_discrete_state(env.reset())
        
        action = agent.start(state)
        while not done:
            new_state, reward, done, _ = env.step(action)
            episode_reward+=reward

            new_state = agent.get_discrete_state(new_state)
            action = agent.step(new_state, reward)
            
        if episode%show_every==0:
            print("episode: {} | mean reward: {}".format(episode, np.mean(train_reward))) 
    
        train_reward.append(episode_reward)
    return agent, train_reward


def test_loop(agent:Agent, env:gym.Env, episodes:int):
    for episode in range(1, episodes):

        done=False
        episode_reward = 0
        state = agent.get_discrete_state(env.reset())
        action = agent.start(state)
        while not done:
            new_state, reward, done, _ = env.step(action)
            episode_reward+=reward
            new_state = agent.get_discrete_state(new_state)
            action = agent.step(new_state, reward)

            env.render()
            time.sleep(0.05)

        
        print("episode: {} | mean reward: {}".format(episode, np.mean(episode_reward))) 

        
env = gym.make("MountainCar-v0")
env.reset()

agent = Agent(env.action_space, env.observation_space)

rewards = []

agent, train_rewards = training_loop(agent=agent, env=env, episodes=10_000, show_every=100)
test_loop(agent, env, 10)
env.close()