from functools import total_ordering
from importlib.metadata import entry_points
import os, sys
import gym
import numpy as np
import random
import tensorflow as tf
sys. path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.GymAgent import Agent

from gym.envs.registration import registry, register

register(
    id = 'FrozenLake-v1',
    entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': "4x4", "is_slippery": False},
    max_episode_steps = 100,
    reward_threshold = 0.70
)

env_name = 'FrozenLake-v1'
env = gym.make(env_name)

class QNAgent(Agent):
    def __init__(self, env, discount_rate=0.97, learning_rate=0.01):
        super().__init__(env)
        self.state_size = env.observation_space.n
        print("State size:", self.state_size)
        
        self.eps = 1.0
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.build_model()
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
    def build_model(self):
        self.state_in = tf.placeholder(tf.int32, shape = [1])
        self.action_in = tf.placeholder(tf.int32, shape = [1])
        self.target_in = tf.placeholder(tf.int32, shape = [1])

        self.state = tf.one_hot(self.state_in, depth=self.state_size)
        self.action = tf.one_hot(self.action_in, depth=self.action_size)

        self.q_state = tf.layers.dense(self.state, units = self.action_size)
        self.q_action = tf.reduce_sum(tf.multiply(self.q_state, self.action), axis = 1)

        self.loss = tf.reduce_sum(tf.square(self.target_in - self.q_action))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def get_action(self, state):
        q_state = self.sess.run(self.q_state, feed_dict = {self.state_in, state})
        action_greedy = np.argmax(q_state)
        action_random = super().get_action(state)
        
        rand_num = random.random()
        if rand_num<self.eps:
            print('random!')
            return action_random
        else:
            return action_greedy
        # return action_random if random.random() < self.eps else action_greedy
    
    def train(self, experience):
        state, action, next_state, reward, done = ([exp] for exp in experience)
        
        q_next = self.sess.run(self.q_state, feed_dict = {self.state_in, next_state})
        q_next[done] = np.zeros([self.action_size])
        q_target = reward + self.discout_rate * np.max(q_next)

        feed = {self.state_in:state, self.action_in : action, self.target_in:q_target}
        self.sees.run(self.optimizer, feed_dict = feed)

        if experience[4]:
            print('============done================')
            self.eps = self.eps * 0.93
    
    # def __del__(self):
    #     self.sess.close()

agent = QNAgent(env)
total_reward = 0

for ep in range(2000):
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        agent.train((state, action, next_state, reward, done))
        state = next_state
        total_reward += reward
        print(f'Episode : {ep}, Total reward : {total_reward}')
        env.render()