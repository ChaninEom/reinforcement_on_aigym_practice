import os, sys
import gym

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.GymAgent import Agent

env_name = 'CartPole-v1'
env = gym.make(env_name)

# print(env.observation_space)
# print(env.action_space)

agent = Agent(env)
state = env.reset()
for i in range(200):
    action = agent.get_action(state)
    state, reward, done, info=env.step(action)
    env.render()