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
        :param env: OpenAI environment. env.P represents the transition probabilities of the environment.
        :param theta: float, threshold for convergence
        :param discount_factor: float, discount factor
        :return: tuple(np.array([n_statesxn_actions]), float), (policy,V) , the optimal policy, and optimal value function
    """
    # Initialize value function
    V = np.zeros(env.nS)
    converged = False
    while not converged:
        # Stopping condition
        delta = 0
        # Go through all states
        for s in range(env.nS):
            # Find the best action
            A = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    A[a] += prob * (reward + discount_factor * V[next_state])
            # Get the value of best action
            best_action_value = np.max(A)
            # Update delta for this state
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update value function
            V[s] = best_action_value
        # Convergence criteria
        if delta < epsilon:
            converged = True


        # Determine optimal policy through value function
        policy = np.zeros([env.nS, env.nA])
        for s in range(env.nS):
            # Gest best actions for each state
            A = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    A[a] += prob * (reward + discount_factor * V[next_state])
            best_action = np.argmax(A)
            # Get optimal policy with an indicator matrix
            policy[s, best_action] = 1.0
        return policy, V


def policy_iteration():
    return

print env.action_space
print env.observation_space

print "Value Iteration: Starting...."
policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

# print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
# print(np.reshape(np.argmax(policy, axis=1), env.shape))
# print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")


