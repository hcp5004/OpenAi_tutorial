#!/usr/bin/env python3

import numpy as np
import gymnasium as gym
from collections import defaultdict
import tilecoding

class mountaincart:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95
            ):
        
        #self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.q_values = 0 # w^T x(s,a)
        self.w = np.zeros((4096,1))
        self.iht = tilecoding.IHT(4096)
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
        self.env = env

    def get_action(self, obs):
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            x = float(obs[0])
            xdot = float(obs[1])
            obs_x_1 = np.zeros(self.w.shape) 
            obs_x_1[tilecoding.tiles(self.iht, 8, [8*x/(0.5+1.2), 8*xdot/(0.07 + 0.07)] , [1])] = 1

            obs_x_2 = np.zeros(self.w.shape) 
            obs_x_2[tilecoding.tiles(self.iht, 8, [8*x/(0.5+1.2), 8*xdot/(0.07 + 0.07)] , [-1])] = 1


            q_value_1 = self.w.T@obs_x_1
            q_value_2 = self.w.T@obs_x_2

        return np.array([1 if q_value_1> q_value_2 else -1])
            #return float(np.argmax(self.q_values[obs]))
    
    def update(
        self,
        obs,
        action,
        reward,
        terminated: bool,
        next_obs,
    ):
        """Updates the Q-value of an action."""
        #future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        x = float(obs[0])
        xdot = float(obs[1])
        
        '''
        from state to feature
        '''
        obs_x = np.zeros(self.w.shape) 
        next_obs_x = np.zeros(self.w.shape)

        obs_x[tilecoding.tiles(self.iht, 8, [8*x/(0.5+1.2), 8*xdot/(0.07 + 0.07)] ,action.tolist() )] = 1
        next_obs_x[tilecoding.tiles(self.iht, 8, [8, 8*x/(0.5+1.2), 8*xdot/(0.07 + 0.07)] ,action.tolist())] = 1

        q_value = self.w.T@obs_x
        future_q_value = (not terminated) * self.w.T@next_obs_x


        temporal_difference = (
            #reward + self.discount_factor * future_q_value - self.q_values[obs][action]
            reward + self.discount_factor * future_q_value - q_value
        )

        gradient = (
            obs_x
        )

        # self.q_values[obs][action] = (
        #     self.q_values[obs][action] + self.lr * temporal_difference
        # )

        self.w = (
            self.w + self.lr * temporal_difference * gradient
        )

        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)