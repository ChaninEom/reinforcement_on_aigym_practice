import gym
import random

env_name = 'MountainCar-v0'
env = gym.make(env_name)

class Agent():
    def __init__(self, env):
        self.action_size = env.action_space.n
        print('Action size :', self.action_size)
    
    def get_action(self, state):
        action = random.choice(range(self.action_size))
        #pole_angle = state[2]
        # action = 0 if pole_angle < 0 else 1
        return action

agent = Agent(env)
state = env.reset()
for i in range(200):
    action = agent.get_action(state)
    observation, reward, donne, info = env.step(action)
    env.render()