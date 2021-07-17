from DeepQLearning import DeepQAgent, training_loop
import gym
from threading import Thread



envs = [] 
agents = []

for i in range(4):
    envs.append(gym.make("MountainCar-v0"))
env = envs[0]

agent = DeepQAgent(name = "agent1", action_space=env.action_space, observation_space=env.observation_space, \
    epsilon=5e-1, epsilon_decay=0.999, epsilon_end=1e-3, discount=0.8, batch_size=64, learning_rate=1e-1)
agents.append(agent)

agent = DeepQAgent(name = "agent2", action_space=env.action_space, observation_space=env.observation_space, \
    epsilon=5e-1, epsilon_decay=0.99, epsilon_end=1e-3, discount=0.8, batch_size=64, learning_rate=1e-2)
agents.append(agent)

agent = DeepQAgent(name = "agent3", action_space=env.action_space, observation_space=env.observation_space, \
    epsilon=1e-1, epsilon_decay=0.9, epsilon_end=5e-3, discount=1, batch_size=64, learning_rate=1e-1)
agents.append(agent)

agent = DeepQAgent(name = "agent4", action_space=env.action_space, observation_space=env.observation_space, \
    epsilon=0.9, epsilon_decay=0.999, epsilon_end=1e-3, discount=0.8, batch_size=256, learning_rate=1e-1)
agents.append(agent)



for i, agent in enumerate(agents):
    x = Thread(target=training_loop, args = (agent, envs[i], 1000, 5))
    x.name=agent.name
    x.start()


    