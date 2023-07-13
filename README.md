# OpenAi_tutorial

This code is implementation of Episodic Semi-gradient Sarsa for Estimating $\hat{q}\approx q_{*}$ for Mountain Car task from the book "Reinforcement learning" by Sutton and Barto.

I divided the total task into four part. 

## agent.py
agent.py is the part for updating w of q estimation and getting action by e-greedy.

## tilecoding.py
tilecoding.py is for tile coding that is one way of the coarse coding to represent state and action to features.

## visualizing.py
visualizing.py is for plotting graph of state value.

## tutorial.py
tutorial.py is the main part of the task including the main pipe line of the algorithm.

## Execution
```
python tutorial.py
```
