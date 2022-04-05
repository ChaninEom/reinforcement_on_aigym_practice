import os, sys
from sre_parse import State
import gym
import numpy as np
import random
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.GymAgent import Agent

env_name = "FrozenLake-v1"
env = gym.make(env_name)

class QAgent(Agent):
    def __init__(self, env, discount_rate = 0.97, learning_rate = 0.01):
        super().__init__(env)
        self.eps = 1.0
        self.state_size = env.observation_space.n
        self.discout_rate = discount_rate
        self.learning_rate = learning_rate
        print('state sizes :', self.state_size)

        self.build_model()
    
    def build_model(self):
        self.q_table = np.random.random((self.state_size, self.action_size))
    
    def get_action(self, state):
        q_state = self.q_table[state]
        action_greedy = np.argmax(q_state)
        action_random = super().get_action(state)
        return action_random if random.random() < self.eps else action_greedy
    
    def train(self, experience):
        state, action, next_state, reward, done = experience
        
        q_next = self.q_table[next_state]
        q_next = np.zeros([self.action_size]) if done else q_next
        q_target = reward + self.discout_rate * np.max(q_next)
        q_update = q_target - self.q_table[state,action]
        self.q_table[state,action] += self.learning_rate * q_update
        
        if done:
            self.eps = self.eps * 0.99

agent = QAgent(env)

total_reward = 0
for ep in range(200):
    state = env.reset()
    done = False
    print('rest')
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        import time
        t=['right', 'left', 'up', 'down']
        print("==================")
        print(t[action])
        print("==================")
        time.sleep(2)
        agent.train((state, action, next_state, reward, done))
        state = next_state
        total_reward += reward
        print(f"Episode : {ep}, Total reward : {total_reward}")
        env.render()