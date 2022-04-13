import os, sys
import gym
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.GymAgent import Agent

env_name = 'MountainCar-v0'
env = gym.make(env_name)

class CarAgent(Agent):
    def __init__(self, env, gamma = 1e-2):
        super().__init__(env)
        self.observation_size = env.observation_space.shape[0]
        self.gamma = gamma

        self.build_model()
    
    def build_model(self):
        self.weights =np.random.randn(self.observation_size, self.action_size)
        self.bias = np.random.randn(self.action_size)
        self.best_weights = self.weights.copy()
        self.psudo_reward = -np.Inf
        self.best_reward = -np.Inf

    def get_action(self, state):
        result = np.dot(state, self.weights)
        action = np.argmax(result)

        return action

    def update(self, position, reward):
        if reward ==0:
            return
        else:
            self.psudo_reward = self.reward_function(position)
            if self.psudo_reward >= self.best_reward:
                self.best_weights = self.weights.copy()
                self.gamma = min(self.gamma/2, 1e-3)
            else:
                self.gamma = max(self.gamma*2/ 2)
            self.weights = self.weights + self.bias*self.gamma
            self.bias = np.random.randn(self.action_size)
            print(self.bias)

    def reward_function(self, position):
        if position <0:
            position /= 2
        
        psudo_reward = abs(position)
        return psudo_reward

agent = CarAgent(env)
state = env.reset()
for ep in range(200):
    for i in range(200):
        action = agent.get_action(state)
        observation, reward, donne, info = env.step(action)
        agent.update(observation[0], reward)
        env.render()