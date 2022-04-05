import os, sys
import gym

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.GymAgent import Agent

env_name = 'MountainCar-v0'
env = gym.make(env_name)

agent = Agent(env)
state = env.reset()
for i in range(200):
    action = agent.get_action(state)
    observation, reward, donne, info = env.step(action)
    env.render()