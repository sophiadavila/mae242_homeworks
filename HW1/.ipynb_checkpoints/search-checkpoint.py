#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sonia martinez
"""

# Please do not distribute or publish solutions to this
# exercise. You are free to use these problems for educational purposes, please refer to the source.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from mazemods import maze
from mazemods import makeMaze
from mazemods import collisionCheck
from mazemods import makePath
from mazemods import getPathFromActions
from mazemods import getCostOfActions
from mazemods import stayWestCost
from mazemods import stayEastCost
import mediumMaze


# defining the maze parameters for medium maze
n = mediumMaze.n
m = mediumMaze.m
O = mediumMaze.O

xI = (1,1)
xG = (9,9)


# # def depthFirstSearch(xI,xG,n,m,O):
# """
#   Search the deepest nodes in the search tree first.
 
#   Your search algorithm needs to return a list of actions
#   and a path that reaches the goal.  
#   Make sure to implement a graph search algorithm.
#   Your algorithm also needs to return the cost of the path. 
#   Use the getCostOfActions function to do this.
#   Finally, the algorithm should return the number of visited
#   nodes in your search.
 
#   """
# "*** YOUR CODE HERE ***"

# Creating maze and plotting it (before implementing search algorithms)
makeMaze(n,m,O)
maze(n,m,O)
plt.show()

#define possible actions
actions = [(1,0), (0,1), (-1,0), (0,-1)]

def depthFirstSearch(xI,xG,n,m,O):
    # Problem state parameters - xI: initial state, xG: goal state
    # Problem Environment Parameters - n: number of rows, m: number of columns, O: list of obstacles
    
    # Initializing the stack and visited set and creating parent map
    stack = [(xI, [])] 
    visited = {xI}
    nodes_visited = 1

    while stack:
        current_state = stack.pop()
        if current_state[0] == xG:
            return current_state[1], stack
            #path = path_from_map(current_state[0], parent_map) #need to create this path from map function
        else:
            for action in actions:
                next_state = (current_state[0] + action[0], current_state[1] + action[1])
                if next_state not in visited and not collisionCheck(current_state, action, O):
                    stack.append((next_state, current_state[1] + [action]))
                    visited.add(next_state)
                    nodes_visited += 1

    return #best_path, cost_path, nodes_visited,plot_path

# # def depthFirstSearch0(xI,xG,n,m,O):
# #     visited = set()
# #     stack = [(xI, [])]
# #     num_visited = 0

# #     while stack:
# #         current, path = stack.pop()
# #         num_visited += 1

# #         if current == xG:
# #             return path, getCostOfActions(xI, path, O), num_visited

# #         if current not in visited:
# #             visited.add(current)

# #             for action in [(1,0), (-1,0), (0,1), (0,-1)]:
# #                 next_state = (current[0] + action[0], current[1] + action[1])
# #                 if not collisionCheck(current, action, O):
# #                     stack.append((next_state, path + [action]))

# #     return None, float('inf'), num_visited



# # def breadthFirstSearch(xI,xG,n,m,O):
# """
#   Search the shallowest nodes in the search tree first [p 85].
 
#   Your search algorithm needs to return a list of actions
#   and a path that reaches the goal. Make sure to implement a graph 
#   search algorithm.
#   Your algorithm also needs to return the cost of the path. 
#   Use the getCostOfActions function to do this.
#   Finally, the algorithm should return the number of visited
#   nodes in your search.

#   """
# "*** YOUR CODE HERE ***"




# # def DijkstraSearch(xI,xG,n,m,O,cost=westCost):
# """
#   Search the nodes with least cost first. 
  
#   Your search algorithm needs to return a list of actions
#   and a path that reaches the goal. Make sure to implement a graph 
#   search algorithm.
#   Your algorithm also needs to return the total cost of the path using
#   either the stayWestCost or stayEastCost function.
#   Finally, the algorithm should return the number of visited
#   nodes in your search.
#   """
# "*** YOUR CODE HERE ***"

# def nullHeuristic(state,goal):
#    """
#    A heuristic function estimates the cost from the current state to the nearest
#    goal.  This heuristic is trivial.

#    """
#    return 0

# #def aStarSearch(xI,xG,n,m,O,heuristic=nullHeuristic):
# "Search the node that has the lowest combined cost and heuristic first."
# """The function uses a function heuristic as an argument. We have used
#   the null heuristic here first, you should redefine heuristics as part of 
#   the homework. 
#   Your algorithm also needs to return the total cost of the path using
#   getCostofActions functions. 
#   Finally, the algorithm should return the number of visited
#   nodes during the search."""
# "*** YOUR CODE HERE ***"
  

    
# # Plots the path
# def showPath(xI,xG,path,n,m,O):
#     gridpath = makePath(xI,xG,path,n,m,O)
#     fig, ax = plt.subplots(1, 1) # make a figure + axes
#     ax.imshow(gridpath) # Plot it
#     ax.invert_yaxis() # Needed so that bottom left is (0,0)
     
# if __name__ == '__main__':
#     # Run test using smallMaze.py (loads n,m,O)
#     from smallMaze import *
#     # from mediumMaze import *  # try these mazes too
#     # from bigMaze import *     # try these mazes too
#     maze(n,m,O) # prints the maze
    
#     # Sample collision check
#     x, u = (5,4), (1,0)
#     testObs = [[6,6,4,4]]
#     collided = collisionCheck(x,u,testObs)
#     print('Collision!' if collided else 'No collision!')
    
#     # Sample path plotted to goal
#     xI = (1,1)
#     xG = (20,1)
#     actions = [(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0),(0,1),
#                (1,0),(1,0),(1,0),(0,-1),(1,0),(1,0),(1,0),(1,0),(1,0),(1,0)]
#     path = getPathFromActions(xI,actions)
#     showPath(xI,xG,path,n,m,O)
    
#     # Cost of that path with various cost functions
#     simplecost = getCostOfActions(xI,actions,O)
#     westcost = stayWestCost(xI,actions,O)
#     eastcost = stayEastCost(xI,actions,O)
#     print('Basic cost was %d, stay west cost was %d, stay east cost was %d' %
#           (simplecost,westcost,eastcost))
    
    
