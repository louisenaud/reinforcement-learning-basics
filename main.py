"""
Project:    
File:       main.py
Created by: louise
On:         6/12/17
At:         4:56 PM
"""

import numpy as np
from gym.envs.toy_text import FrozenLakeEnv, CliffWalkingEnv
import time

env = FrozenLakeEnv()

def value_iteration(env, theta=0.0000001, discount_factor=0.99):
    """
        Value Iteration Algorithm.
        The notations are from Reinforcement Learning: An Introduction, by Sutton et al.
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
                for p, next_state, rt, flag in env.P[s][a]:  # Compute each action value
                    actions[a] += p * (rt + discount_factor * values[next_state])
            # Get the value of best action
            best_action_value = np.max(actions)
            # Update delta for this state
            delta = max(delta, np.abs(best_action_value - values[s]))
            # Update value function
            values[s] = best_action_value
        # Convergence criteria
        if delta < theta:
            converged = True

    # Determine optimal policy through value function
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # Get best actions for each state
        actions = np.zeros(env.nA)
        for a in range(env.nA):
            for p, next_state, rt, flag in env.P[s][a]:
                actions[a] += p * (rt + discount_factor * values[next_state])
        best_action = np.argmax(actions)
        # Get optimal policy with an indicator matrix
        policy[s, best_action] = 1.0
    return policy, values


def policy_iteration(env, theta=0.000001, discount_factor=0.9999):
    """
    Policy Iteration Algorithm. Alternates Policy Evaluation Step with Policy Improvement Step.
    The notations are from Reinforcement Learning: An Introduction, by Sutton et al.
    :param env: OpenAI environment. In this environment, env.P is a dictionary with two keys - state, action- that 
                    contains the transition probabilities of the environment, the next state and the reward for each 
                    possible pair (state, action) in a tuple.
    :param theta: float, threshold for convergence
    :param discount_factor: float, discount factor, should be <1 for convergence.
    :return: tuple(np.array([n_states x n_actions]), float), (policy,V) , the optimal policy, and optimal value function

    """
    # Set initial policy
    policy = np.ones([env.nS, env.nA]) / (env.nA * env.nS)
    while True: # As long as the policy is the same as the previous one, we haven't converged.
        # Policy Evaluation Step
        ##################################
        # Set Value function to 0
        V = np.zeros(env.nS)
        converged = False
        while not converged:
            delta = 0
            for s in range(env.nS):  # For each state compute and store Value
                v = 0
                for a, action_probability in enumerate(policy[s]):
                    for p, next_state, rt, flag in env.P[s][a]:
                        # Compute Value
                        v += action_probability * p * (rt + discount_factor * V[next_state])
                # Update Delta for this state
                delta = max(delta, np.abs(v - V[s]))
                V[s] = v
            # When the Value Function doesn't change much anymore, we have converged for the policy evaluation step
            if delta < theta:
                converged = True

        # Policy Improvement Step
        ##################################
        # Flag to check convergence introduced in Sutton et al.
        policy_stable = True

        # Go through all states
        for s in range(env.nS):
            # Get the best action for this state under current policy
            a_opt = np.argmax(policy[s])

            # Get the action that maximizes the value
            action_values = np.zeros(env.nA)
            for a in range(env.nA):  # Computes Action Values in order to obtain the best action
                for p, next_state, rt, flag in env.P[s][a]:
                    action_values[a] += p * (rt + discount_factor * V[next_state])
            # Compute the greedy policy
            best_a = np.argmax(action_values)
            if a_opt != best_a:  # Check if the policy from previous step is different from the current one
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]  # Update the current policy for next Policy Evaluation step

        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V


t_start = time.time()
print "Value Iteration: Starting...."
policy, v = value_iteration(env)
t_end = time.time()

print "Executed in ", t_end - t_start, "seconds"
print "Policy:"
print policy

print "Value Function:"
print v

t_start = time.time()

print "Policy Iteration: Starting...."
policy, v = policy_iteration(env)
t_end = time.time()

print "Executed in ", t_end - t_start, "seconds"
print "Policy:"
print policy

print "Value Function:"
print v



