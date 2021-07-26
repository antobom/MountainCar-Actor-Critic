import matplotlib.pyplot as plt
from collections import deque

class Plotter:
    def __init__(self, mean_reward_len):
        self.t = []
        self.reward = []

        self.t_mean_reawrd = []
        self.mean_reward = []

        self.cpt = 0
        self.mean_reward_len = mean_reward_len

    def update_plot(self, new_reward):
        self.t.append(self.cpt)
        self.reward.append(new_reward)
        if self.cpt > self.mean_reward_len:
            self.mean_reward.append(sum(self.reward[-self.mean_reward_len:]) / self.mean_reward_len)
            self.t_mean_reawrd.append(self.cpt)
        self.cpt+=1

        plt.plot(self.t, self.reward, 'r-', self.t_mean_reawrd, self.mean_reward, 'g-')
        plt.pause(0.0001)