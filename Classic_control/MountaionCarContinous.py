import gym
import random

env_name = 'MountainCarContinuous-v0'
env = gym.make(env_name)


class Agent():
    def __init__(self, env):
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        print('Action low :', self.action_low)
        print('Action high :', self.action_high)
    def get_action(self, state):
        action = random.uniform(self.action_low, self.action_high)
        return action

agent = Agent(env)
state = env.reset()

for i in range(200):
    action = agent.get_action(state)
    state, reward, done, info = env.step(action)
    env.render()