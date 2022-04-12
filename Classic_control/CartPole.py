from math import gamma
import os, sys
import gym
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.GymAgent import Agent

env_name = 'CartPole-v1'
env = gym.make(env_name)

class PoleAgent(Agent):
    def __init__(self, env, gamma = 1e-2):
        super().__init__(env)
        self.state_size = env.observation_space.shape[0]
        self.gamma = gamma

        self.build_model()

    def build_model(self):
        self.weight = np.random.random((self.state_size, self.action_size))
        self.bias = np.random.random((1, self.action_size))*1e-2
        self.best_weight = self.weight.copy()
        self.best_reward = -np.Inf

    def get_action(self, state):
        result = np.dot(state, self.weight)+self.bias
        action = np.argmax(result)
        return action
    
    def update(self, reward):
        if reward >= self.best_reward:
            self.best_weight = self.weight.copy()
            self.best_reward = reward
            self.gamma = min(self.gamma/2, 1e-3)
        else:
            self.gamma = max(self.gamma*2, 2)

        self.bias =  np.random.random((1, self.action_size))*self.gamma
        self.weight = self.weight + self.bias


agent = PoleAgent(env)
for ep in range(500):
    state = env.reset()
    total_reward = 0
    for i in range(200):
        action = agent.get_action(state)
        state, reward, done, info=env.step(action)
        agent.update(reward)
        env.render()
        total_reward += reward
    print(f"episode : {ep}, total_reward : {total_reward}")