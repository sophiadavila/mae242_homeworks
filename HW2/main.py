#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 09:12:40 2022

@author: sonia, parth, yang

Edited by Sophia Davila for HW2, Spring 2026 - May 8th 2026
"""

import numpy as np
import matplotlib.pyplot as plt

from grids import SmallGrid, MediumGrid, MediumGridBridge

############################################################################
# Part 1: In this first part, we show you how to calculate the
# transition probabilities for a given grid and error
# probability. Then we show how to compute the expected value of
# taking an action `a` in state `s`. Both of these functions will be
# helpful in implementing value iteration and policy iteration.
############################################################################

#print("Hello World")

def transitionMatrix(grid, eta):
    """Compute the transition matrix `P` for all states in `grid` with
    action error rate `eta`.

    `P` is a numpy array, indexed such that `P[s_p][s][a]` is the
    probability of transitioning to `s_p` from `s` under action `a`.

    """

    # Transition matrix will be indexed by ((i_s_p, j_s_p), (i_s, j_s), a)
    P = np.zeros((grid.n, grid.m, grid.n, grid.m, grid.n_a))

    # Note that numpy allows different styles of indexing. If we
    # define the tuples s_p = (i_s_p, j_s_p) and s = (i_s, j_s), then
    # the expression P[s_p][s] is equivalent to P[i_s_p][j_s_p][i_s][j_s].
    # We use this to make the following code more readable and concise.

    for s in grid.states:
        # Assign the transition probabilities for each input
        for a in range(grid.n_a):
            # Nominal case, no error is introduced.
            s_p = grid.nextState(s, grid.u_set[a])
            P[s_p][s][a] += 1 - eta

            # Error case, rotation to the left
            s_p = grid.nextState(s, grid.u_set[(a + 1) % grid.n_a])
            P[s_p][s][a] += eta / 2

            # Error case, rotation to the right
            s_p = grid.nextState(s, grid.u_set[(a - 1) % grid.n_a])
            P[s_p][s][a] += eta / 2

    return P


def expectedValue(P, V, s, a, gamma, grid):
    """Compute the expected value of taking an action from a given state.

    Inputs:
    `P`: Matrix of transition probabilities
    `V`: Value function (e.g., the estimate from the current iteration)
    `s`: State
    `a`: Action
    `gamma`: Discount factor
    `grid`: The grid world

    Outputs: The inner term of the Bellman operator, i.e., sum over
    all s' of P(s' | s, a)*(R(s, a, s') + gamma*V(s')). Note that in
    this problem the reward only depends on s' and is the negative of
    the cost.

    """
    # Get the set of all possible next states
    s_primes = set(grid.nextState(s, u) for u in grid.u_set)

    # Sum over them to compute the expectation
    return sum(P[s_p][s][a] * (-grid.cost(s_p) + gamma*V[s_p])
               for s_p in s_primes)


############################################################################
# Part 2: Now it's time to implement value iteration and policy
# iteration. We have provided templates that show the basic structure
# your code should follow.
############################################################################


def valueIteration(gamma, eta, grid):
    """Implement value iteration with a discount factor `gamma` and
    noise probability `eta`.

    Outputs:
    `V`: Value function, numpy array of (n,m) dimensions
    `pi`: Policy, numpy array of (n,m) dimensions

    """
    P = transitionMatrix(grid, eta)

    tolerance = 1e-3                            # Convergence error

    V = np.zeros([grid.n, grid.m])              # Value function - tested different value function initializations for part (b)
    #V = np.ones([grid.n, grid.m])
    #V = np.random.rand(grid.n, grid.m)
    #V = -10*np.ones([grid.n, grid.m])

    pi = np.zeros([grid.n, grid.m], dtype=int)  # Policy
    iterations = 0                              # No. of iterations

    # ------- Your code goes here -------
    # VALUE ITERATION
    # while the difference between the new and old value function is larger than the tolerance, 
    # this value iteration will continue to update the value function V and policy pi. 
    # This uses expectedValue function to compute V and update pi by choosing the action 
    # that maximizes the expected value.Once the value function converges, we return the value 
    # function, policy, and number of iterations.

    # defining all the variables used ->
    # delta is the difference between the new and old value function that checks for convergence
    # old_v is the value function before being updated.
    # best_value is the current best value so far for a state. It updtated as we loop through actions
    # a are actions
    # a_value is the expected value for a certain action
    # best action is the action that gives the best value
    # V is the value function for a state s (updated)
    # pi is the policy, which contains a list of actions for each state
    # iterations is the number of iterations for value iteration

    delta = 10000 #initializing delta to be larger than tolerance so that the while loop starts
    while delta > tolerance:
        delta = 0
        for s in grid.states:
            old_v = V[s]
            best_value = -10000 #initializing best value to be a very small number so that it will be updated by the first action value
            
            #the following for loop is equivalent to the max over a of the expected value. It will calculate the best value and the best action
            #to achieve this best value.
            for a in range(grid.n_a):
                a_value = expectedValue(P, V, s, a, gamma, grid)
                if a_value > best_value:
                    best_value = a_value
                    best_action = a
            V[s] = best_value 
            pi[s] = best_action # already extracts the best action for each state, so we can update the policy at the same time as we update the value function.
            delta = max(delta, abs(old_v - V[s]))
            # outputting the best policy

        iterations += 1

    return V, pi, iterations


def policyIteration(gamma, eta, grid):
    """Implement (offline) policy iteration with a discount factor
    `gamma` and noise probability `eta`.

    Output:
    `V`: Value function, numpy array of (n,m) dimensions
    `pi`: Policy, numpy array of (n,m) dimensions

    """
    P = transitionMatrix(grid, eta)

    tolerance = 1e-3                            # Tolerance for evaluation
    V = np.zeros([grid.n, grid.m])              # Value function
    pi = np.zeros([grid.n, grid.m], dtype=int)  # Policy
    i_imprv = 0                                 # No. of improvement iterations
    i_eval = 0                                  # No. of evaluation iterations

    # ------- Your code goes here -------
    # Start large while loop
    policy_stable = False
    while not policy_stable:
        # POLICY EVALUATION
        # start with policy evaluation, then policy improvement, and repeat until convergence
        # policy evaluation: while the difference between the new and old value function is larger than the tolerance,
        # this policy evaluation will continue to update the value function V. This uses expectedValue function to compute V.
        
        # defining all the variables used ->
        # delta is the difference between the new and old value function that checks for convergence
        # old_v is the value function before being updated.
        # curr_a is the current action for a state under the current policy
        # V is the value function for a state s (updated)
        # i_eval is the number of iterations for policy evaluation

        delta = 10000 #initializing delta to be larger than tolerance so that the while loop starts
        while delta > tolerance:
            delta = 0
            for s in grid.states:
                old_v = V[s]
                curr_a = pi[s]
                V[s] = expectedValue(P, V, s, curr_a, gamma, grid)
                delta = max(delta, abs(old_v - V[s]))
            i_eval += 1
        
        # POLICY IMPROVEMENT
        # now we move into policy improvement, where it will update the policy by choosing what action maximises the expected value.
        # It will choose the new policy/action as the action that maximizes the value. It will check if the policy changed at all.
        # if the policy changed, they the policy is set as not stable, and the while loop continues from policy evaluation again. 
        # if the policy did not change, we end the while loop and return the value function and policy.

        # old_a is the action for a state under the old policy that will be compared to the other policies
        # best_value is the current best value so far for a state. It updtated as we loop through actions
        # a are actions
        # a_value is the expected value for a certain action
        # best action is the action that gives the best value
        # pi is the policy, which contains a list of actions for each state
        # policy stable defines whether the policy has changed or not, which is used to determine whether to continue the while loop or not.

        for s in grid.states:
            old_a = pi[s]
            best_value = -10000 #initializing best value to be a very small number so that it will be updated by the first action value
            for a in range(grid.n_a):
                a_value = expectedValue(P, V, s, a, gamma, grid)
                if a_value > best_value:
                    best_value = a_value
                    best_action = a
            pi[s] = best_action
            if old_a != best_action:
                policy_stable = False
            else:
                policy_stable = True
        i_imprv += 1

    return V, pi, i_eval, i_imprv


############################################################################
# Part 3: Here is where you will run your value and policy iteration
# algorithms, following the instructions from the pdf. We have
# provided an outline of what functions you might call to help you
# answer the questions.
############################################################################


# Parameters for each subproblem, stored as [gamma, eta]
params = {
    'a': [0.9, 0.2],
    'b': [0.9, 0.2],
    # For parts c and d, fill in the values of gamma and eta that
    # result in the desired behavior
    'c': [0.9, 0.2],
    'd1': [0.3, 0.01],
    'd2': [0.2, 0.2],
    'd3': [0.6, 0.1],
    'd4': [0.4, 0.1],
}

# viewing small grid
smallGrid = SmallGrid()
#smallGrid.plot(show=True)
# viewing medium grid
mediumGrid = MediumGrid()
#mediumGrid.plot(show=True)
# viewing medium grid with bridge
mediumGridBridge = MediumGridBridge()
#mediumGridBridge.plot(show=True)

def plotting_small_stuff(V, pi):
    if True:  # Change this to true to see the output
        # Example policy and value function
        # V = np.random.rand(mediumGrid.n, mediumGrid.m)
        # pi = np.ones((mediumGrid.n, mediumGrid.m), dtype=int)  # Always up

        # Plot the empty grid with start, goal, and penalty locations marked
        # To plot on the small grid instead, use smallGrid.plot
        smallGrid.plot(show=True)

        # Print a value function or policy to the terminal
        smallGrid.printValues(V)
        smallGrid.printPolicy(pi)

        # Plot the value function (red is low, green is high)
        smallGrid.plotValues(V, show=True)

        # Plot a noise-free path from the start node under policy pi
        # Note that these paths won't look good until you compute a good policy
        path = smallGrid.getNominalPathFromPolicy(pi)
        smallGrid.plotPath(path, show=True)

        # Plot a noisy path from the start node under policy pi
        path = smallGrid.getRandomPathFromPolicy(pi, eta)
        smallGrid.plotPath(path, show=True)

def plotting_medium_stuff(V, pi):
    if True:  # Change this to true to see the output
        # Example policy and value function
        # V = np.random.rand(mediumGrid.n, mediumGrid.m)
        # pi = np.ones((mediumGrid.n, mediumGrid.m), dtype=int)  # Always up

        # Plot the empty grid with start, goal, and penalty locations marked
        # To plot on the small grid instead, use smallGrid.plot
        mediumGrid.plot(show=True)

        # Print a value function or policy to the terminal
        mediumGrid.printValues(V)
        mediumGrid.printPolicy(pi)

        # Plot the value function (red is low, green is high)
        mediumGrid.plotValues(V, show=True)

        # Plot a noise-free path from the start node under policy pi
        # Note that these paths won't look good until you compute a good policy
        path = mediumGrid.getNominalPathFromPolicy(pi)
        mediumGrid.plotPath(path, show=True)

        # Plot a noisy path from the start node under policy pi
        path = mediumGrid.getRandomPathFromPolicy(pi, eta)
        mediumGrid.plotPath(path, show=True)

def plotting_mediumBridge_stuff(V, pi):
    if True:  # Change this to true to see the output
        # Example policy and value function
        # V = np.random.rand(mediumGrid.n, mediumGrid.m)
        # pi = np.ones((mediumGrid.n, mediumGrid.m), dtype=int)  # Always up

        # Plot the empty grid with start, goal, and penalty locations marked
        # To plot on the small grid instead, use smallGrid.plot
        mediumGridBridge.plot(show=True)

        # Print a value function or policy to the terminal
        mediumGridBridge.printValues(V)
        mediumGridBridge.printPolicy(pi)

        # Plot the value function (red is low, green is high)
        mediumGridBridge.plotValues(V, show=True)

        # Plot a noise-free path from the start node under policy pi
        # Note that these paths won't look good until you compute a good policy
        path = mediumGridBridge.getNominalPathFromPolicy(pi)
        mediumGridBridge.plotPath(path, show=True)

        # Plot a noisy path from the start node under policy pi
        path = mediumGridBridge.getRandomPathFromPolicy(pi, eta)
        mediumGridBridge.plotPath(path, show=True)

if __name__ == '__main__':

    # ---------- Part (a) ----------- #

    gamma, eta = params['a']
    V, pi, i_eval, i_imprv = policyIteration(gamma, eta, smallGrid)
    
    print(" ")
    print(f"Part (a): {i_eval} evaluations and {i_imprv} improvements.")
    print(" ")

    plotting_small_stuff(V, pi)
    print(" ")


    # ---------- Part (b) ----------- #

    gamma, eta = params['b']
    V, pi, iterations = valueIteration(gamma, eta, smallGrid)

    print(" ")
    print(f"Part (b): {iterations} iterations.")
    print(" ")

    plotting_small_stuff(V, pi)
    print(" ")

    print("The number of iterations changes slightly as we change the initialization of the value function. That happens because different initializations start at different distances from the actual optimal value function.")
    print(" ")
    print("The policy is not the same as in part (a). Although, for this case, the path chosen ends up being the same as in part (a), whe looking at the policy grid, we see that the ploicy is not the same throughout the entire grid.")
    print(" ")

    # ---------- Part (c) ----------- #
    # Change the values in `params` above to get the desired behavior

    gamma, eta = params['c']
    V, pi, _ = valueIteration(gamma, eta, mediumGrid)

    print(" ")
    print(f"Part (c): gamma: {gamma} , eta: {eta}.")
    print(" ")

    plotting_medium_stuff(V, pi)
    print(" ")

    print("The original parameters gama =0.9 and eta = 0.2 already gave me a policy that passes through (2,2). However, this paremeter also ends up reaching the penalty states some times." \
    "Playing around with the parameter values, I found that smaller values of eta tend to favor shorter paths because the actions are more reliable. " \
    "As I increse eta, the policy becomes more conservative and chooses longer paths that avoid the states near failure, since action noise increases " \
    "the probability of accidentally getting to one of the red blovks (penalty states).")
    print(" ")


    # ---------- Part (d) ----------- #
    # Change the values in `params` above to get the desired behavior
    mediumGridBridge = MediumGridBridge()

    # 1) Close exit, risking the cliff
    gamma, eta = params['d1']
    print(" ")
    print(f"Part (d) 1): gamma: {gamma}, eta: {eta}.")
    print(" ")
    V, pi, _ = valueIteration(gamma, eta, mediumGridBridge)
    plotting_mediumBridge_stuff(V, pi)
    print(" ")

    # 2) Close exit, avoiding the cliff
    gamma, eta = params['d2']
    print(" ")
    print(f"Part (d) 2): gamma: {gamma}, eta: {eta}.")
    print(" ")
    V, pi, _ = valueIteration(gamma, eta, mediumGridBridge)
    plotting_mediumBridge_stuff(V, pi)
    print(" ")

    # 3) Far exit, risking the cliff
    gamma, eta = params['d3']
    print(" ")
    print(f"Part (d) 3): gamma: {gamma}, eta: {eta}.")
    print(" ")
    V, pi, _ = valueIteration(gamma, eta, mediumGridBridge)
    plotting_mediumBridge_stuff(V, pi)
    print(" ")

    # 4) Far exit, avoiding the cliff
    gamma, eta = params['d4']
    print(" ")
    print(f"Part (d) 4): gamma: {gamma}, eta: {eta}.")
    print(" ")
    V, pi, _ = valueIteration(gamma, eta, mediumGridBridge)
    plotting_mediumBridge_stuff(V, pi)
    print(" ")

############################################################################
# Part 4: In this final part we demonstrate some of the plotting and
# printing functions available for visualizing and debugging the
# results of your algorithms.
############################################################################

    #def plotting_medium_stuff(V, pi):
    if False:  # Change this to true to see the output
            # Example policy and value function
            # V = np.random.rand(mediumGrid.n, mediumGrid.m)
            # pi = np.ones((mediumGrid.n, mediumGrid.m), dtype=int)  # Always up

            # Plot the empty grid with start, goal, and penalty locations marked
            # To plot on the small grid instead, use smallGrid.plot
            mediumGrid.plot(show=True)

            # Print a value function or policy to the terminal
            mediumGrid.printValues(V)
            mediumGrid.printPolicy(pi)

            # Plot the value function (red is low, green is high)
            mediumGrid.plotValues(V, show=True)

            # Plot a noise-free path from the start node under policy pi
            # Note that these paths won't look good until you compute a good policy
            path = mediumGrid.getNominalPathFromPolicy(pi)
            mediumGrid.plotPath(path, show=True)

            # Plot a noisy path from the start node under policy pi
            path = mediumGrid.getRandomPathFromPolicy(pi, eta)
            mediumGrid.plotPath(path, show=True)

