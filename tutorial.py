#!/usr/bin/env python3

import gymnasium as gym
import tilecoding
import matplotlib.pyplot as plt
import numpy as np
from agent import mountaincart
from tqdm import tqdm
from visualizing import visualizing, policy

'''
Training Loop for Semi-gradient Sarsa for estimating
'''

# hyperparameters
learning_rate = 0.05
n_episodes = 500#100_000
start_epsilon = 0.1
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

#env = gym.make("MountainCarContinuous-v0")
env = gym.make("MountainCar-v0")
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)

agent = mountaincart(
    env = env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

for episode in tqdm(range(n_episodes)):
    obs, info = agent.env.reset()
    
    done = False
    # play one episode
    action = agent.get_action(obs)

    while not done:
        
         # epsilon-greedy
        next_obs, reward, terminated, truncated, info = agent.env.step(action) 
        
        # update the agent
        next_action = agent.get_action(next_obs)
        agent.update(obs, action, reward, terminated, next_obs, next_action)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs
        action = next_action
    

    #agent.decay_epsilon()

visualizing(env= env, agent= agent)
policy(env= env, agent= agent)
plt.show()
env.close()

env = gym.make("MountainCar-v0", render_mode = "human")
agent.epsilon = 0
obs, info = env.reset()

done = False
# play one episode
while not done:
    action = agent.get_action(obs) # epsilon-greedy
    next_obs, reward, terminated, truncated, info = env.step(action) 
    
    # update if the environment is done and the current obs
    done = terminated or truncated
    obs = next_obs

env.close()


