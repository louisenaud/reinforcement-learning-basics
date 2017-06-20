"""
Project:    
File:       main.py
Created by: louise
On:         6/12/17
At:         4:56 PM
"""

import numpy as np
import gym
from gym.envs.toy_text import FrozenLakeEnv, CliffWalkingEnv

env = FrozenLakeEnv()

def value_iteration(env, epsilon=0.001, discount_factor=0.9999):
    """
        Value Iteration Algorithm.
        :param env: OpenAI environment. In this environment, env.P is a dictionary with two keys - state, action- that 
                    contains the transition probabilities of the environment, the next state and the reward for each 
                    possible pair (state, action) in a tuple.
        :param epsilon: float, threshold for convergence
        :param discount_factor: float, discount factor, should be <1 for convergence.
        :return: tuple(np.array([n_states x n_actions]), float), (policy,V) , the optimal policy, and optimal value function
    """
    # Initialize value function
    values = np.zeros(env.nS)
    converged = False
    while not converged:
        # Stopping condition
        delta = 0
        # Go through all states
        for s in range(env.nS):  # Go through all states
            # Find the best action
            actions = np.zeros(env.nA)  # Initialize actions vector
            for a in range(env.nA):  # Go through all actions
                for prob, next_state, reward, done in env.P[s][a]:  # Compute each action value
                    actions[a] += prob * (reward + discount_factor * values[next_state])
            # Get the value of best action
            best_action_value = np.max(actions)
            # Update delta for this state
            delta = max(delta, np.abs(best_action_value - values[s]))
            # Update value function
            values[s] = best_action_value
        # Convergence criteria
        if delta < epsilon:
            converged = True

        # Determine optimal policy through value function
        policy = np.zeros([env.nS, env.nA])
        for s in range(env.nS):
            # Gest best actions for each state
            actions = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    actions[a] += prob * (reward + discount_factor * values[next_state])
            best_action = np.argmax(actions)
            # Get optimal policy with an indicator matrix
            policy[s, best_action] = 1.0
        return policy, values


def policy_iteration():
    return

print env.action_space
print env.observation_space

print "Value Iteration: Starting...."
policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Value Function:")
print(v)
print("")



