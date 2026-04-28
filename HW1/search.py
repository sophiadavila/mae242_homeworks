#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: sonia martinez
#Homework 1 - Sophia Davila
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

import heapq

# defining the maze parameters for medium maze
n = mediumMaze.n
m = mediumMaze.m
O = mediumMaze.O

# defining the initial and goal states for tests
xI = (1,1)
#xG = (30,16)
xG = (10,10)
#xG = (9,9)

# Creating maze and plotting it (before implementing search algorithms)
makeMaze(n,m,O)
maze(n,m,O)
plt.title("Medium Maze")
plt.show()


def depthFirstSearch(xI,xG,n,m,O):
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
  actions = [(1,0), (0,1), (-1,0), (0,-1)]

    # Problem state parameters - xI: initial state, xG: goal state
    # Problem Environment Parameters - n: number of rows, m: number of columns, O: list of obstacles
    
    # Initializing the stack and visited set and creating parent map
  stack = [(xI, [])] 
  visited_list = {xI}
  nodes_visited = 1

  while stack:
      current_state, action_list = stack.pop()
      if current_state == xG:
          path = getPathFromActions(xI, action_list) #path = path_from_actions(xi, action list) #need to create this path from map function
          cost_path = getCostOfActions(xI,action_list,O)
          path_to_plot = makePath(xI,xG,path,n,m,O)
          plt.imshow(path_to_plot)
          plt.gca().invert_yaxis()
          plt.title("Depth First Search Path")
          plt.show()
          #print(visited_list)
          print("Depth First Search:")
          print("The cost of the path is:", cost_path)
          print(nodes_visited," nodes were visited")
          return action_list, cost_path, nodes_visited
      else:
          for action in actions:
              next_state = (current_state[0] + action[0], current_state[1] + action[1])
              if next_state not in visited_list and not collisionCheck(current_state, action, O):
                  stack.append((next_state, action_list + [action]))
                  visited_list.add(next_state)
                  nodes_visited += 1
  if not np.stack:
      print("Could not find a path to goal")
      print("Current state:", current_state)
      print(nodes_visited, " nodes visited")
        #print("List of visited nodes", visited_list)
        #print("List of actions till this point", action_list)

  return action_list, None, nodes_visited

## testing depth first search
print("Testing different algorithms on medium maze. Initial state is (1,1) and goal state is (10,10).")
print()
action_list, cost_path, nodes_visited = depthFirstSearch(xI,xG,n,m,O)
print()
print("Question: Is the exploration order what you would have expected?")
print("Answer: Yes, the algorithm explores in the form 'last in first out', so it explores the most recently added nodes first.")
print()
print("Question:Is this a least cost solution?")
print("Answer: No, depth first search does not guarantee the least cost solution. The algorithm explores as deep as possible along each branch before backtracking, which can lead to less optimal paths if the goal is found in a later branch. (that can be observed by changing the order of the actions and seeing that the path changes)")
print()

def breadthFirstSearch(xI,xG,n,m,O):
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
    # Problem state parameters - xI: initial state, xG: goal state
    # Problem Environment Parameters - n: number of rows, m: number of columns, O: list of obstacles
    
    # Initializing the Q and visited set and creating parent map
    actions = [(1,0), (0,1), (-1,0), (0,-1)]
    Q = [(xI, [])] 
    visited_list = {xI}
    nodes_visited = 1

    while Q:
        current_state, action_list = Q.pop(0)
        if current_state == xG:
            path = getPathFromActions(xI, action_list) #path = path_from_actions(xi, action list) #need to create this path from map function
            cost_path = getCostOfActions(xI,action_list,O)
            path_to_plot = makePath(xI,xG,path,n,m,O)
            plt.imshow(path_to_plot)
            plt.gca().invert_yaxis()
            plt.title("Breadth First Search Path")
            plt.show()
            #print(visited_list)
            print("Breadth First Search:")
            print("The cost of the path is:", cost_path)
            print(nodes_visited," nodes were visited")
            return action_list, cost_path, nodes_visited
        else:
            for action in actions:
                next_state = (current_state[0] + action[0], current_state[1] + action[1])
                if next_state not in visited_list and not collisionCheck(current_state, action, O):
                    Q.append((next_state, action_list + [action]))
                    visited_list.add(next_state)
                    nodes_visited += 1
    if not Q:
        print("Could not find a path to goal")
        print("Current state:", current_state)
        print(nodes_visited, " nodes visited")
        #print("List of visited nodes", visited_list)
        #print("List of actions till this point", action_list)

    return action_list, None, nodes_visited

## testing breadth first search
action_list, cost_path, nodes_visited = breadthFirstSearch(xI,xG,n,m,O)
print()

def DijkstraSearch(xI,xG,n,m,O,cost):
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
    # Problem state parameters - xI: initial state, xG: goal state
    # Problem Environment Parameters - n: number of rows, m: number of columns, O: list of obstacles
    # Cost Function
    # Initializing the Q and visited set and creating parent map
    actions = [(1,0), (0,1), (-1,0), (0,-1)]
    Q = [(0, xI, [])] 
    visited_list = {xI}
    nodes_visited = 1

    while Q:
        cost_so_far, current_state, action_list = heapq.heappop(Q)
        if current_state == xG:
            path = getPathFromActions(xI, action_list) #path = path_from_actions(xi, action list) #need to create this path from map function
            cost_path = cost_so_far
            #cost_path = getCostOfActions(xI,action_list,O)
            path_to_plot = makePath(xI,xG,path,n,m,O)
            plt.imshow(path_to_plot)
            plt.title("Dijkstra's Search Path with " + cost.__name__)
            plt.gca().invert_yaxis()
            plt.show()
            #print(visited_list)
            print("Dijkstra's Search" + cost.__name__ + ":")
            print("The cost of the path is:", cost_path)
            print(nodes_visited," nodes were visited")
            return action_list, cost_path, nodes_visited
        else:
            for action in actions:
                next_state = (current_state[0] + action[0], current_state[1] + action[1])
                if next_state not in visited_list and not collisionCheck(current_state, action, O):
                    new_cost = cost(xI,(action_list + [action]),O)
                    Q.append((new_cost, next_state, action_list + [action]))
                    visited_list.add(next_state)
                    nodes_visited += 1
    if not Q:
        print("Could not find a path to goal")
        print("Current state:", current_state)
        print(nodes_visited, " nodes visited")
        
    return action_list, None, nodes_visited

## testing Dijkstra's search
action_list, cost_path, nodes_visited = DijkstraSearch(xI,xG,n,m,O,cost=stayWestCost)
print()
action_list, cost_path, nodes_visited = DijkstraSearch(xI,xG,n,m,O,cost=stayEastCost)
print()

def nullHeuristic(state,goal):
#    """
#    A heuristic function estimates the cost from the current state to the nearest
#    goal.  This heuristic is trivial.
#    """
  return 0

def manhattanHeuristic(x, xG):
    return sum(abs(a - b) for a, b in zip(x, xG))
    
def euclideanHeuristic(x, xG):
    squared_diffs = sum((a - b) ** 2 for a, b in zip(x, xG))
    return squared_diffs ** 0.5


def aStarSearch(xI,xG,n,m,O,heuristic):
# "Search the node that has the lowest combined cost and heuristic first."
# """The function uses a function heuristic as an argument. We have used
#   the null heuristic here first, you should redefine heuristics as part of 
#   the homework. 
#   Your algorithm also needs to return the total cost of the path using
#   getCostofActions functions. 
#   Finally, the algorithm should return the number of visited
#   nodes during the search."""
# "*** YOUR CODE HERE ***"
    # Problem state parameters - xI: initial state, xG: goal state
    # Problem Environment Parameters - n: number of rows, m: number of columns, O: list of obstacles
    # Cost Function
    # Initializing the Q and visited set and creating parent map
    actions = [(1,0), (0,1), (-1,0), (0,-1)]
    Q = [(0, xI, [])] 
    visited_list = {xI}
    nodes_visited = 1

    while Q:
        predicted_cost, current_state, action_list = heapq.heappop(Q)
        if current_state == xG:
            path = getPathFromActions(xI, action_list)
            cost_path = getCostOfActions(xI,action_list,O) #regular cost - once we reach xG we just calculate it as usual
            path_to_plot = makePath(xI,xG,path,n,m,O)
            plt.imshow(path_to_plot)
            plt.title("A* Search Path with " + heuristic.__name__)
            plt.gca().invert_yaxis()
            plt.show()
            #print(visited_list)
            print("A* Search with " + heuristic.__name__ + ":")
            print("The cost of the path is:", cost_path)
            print(nodes_visited," nodes were visited")
            return action_list, cost_path, nodes_visited
        else:
            for action in actions:
                next_state = (current_state[0] + action[0], current_state[1] + action[1])
                if next_state not in visited_list and not collisionCheck(current_state, action, O):
                    predicted_cost = getCostOfActions(xI,action_list,O) + heuristic(current_state, xG)
                    #print(predicted_cost)
                    Q.append((predicted_cost, next_state, action_list + [action]))
                    visited_list.add(next_state)
                    nodes_visited += 1
    if not Q:
        print("Could not find a path to goal")
        print("Current state:", current_state)
        print(nodes_visited, " nodes visited")
        #print("List of visited nodes", visited_list)
        #print("List of actions till this point", action_list)

    return action_list, None, nodes_visited
   
## testing A*
action_list, cost_path, nodes_visited = aStarSearch(xI,xG,n,m,O,heuristic = manhattanHeuristic)
print()
action_list, cost_path, nodes_visited = aStarSearch(xI,xG,n,m,O,heuristic = euclideanHeuristic)
print()
action_list, cost_path, nodes_visited = aStarSearch(xI,xG,n,m,O,heuristic = nullHeuristic)
print()
print("Question: Which heuristic is superior?")
print("Answer: The Manhattan heuristic is superior in this case because it provides a more accurate estimate of the distance to the goal in a grid-based environment, leading to fewer nodes being explored and a more efficient search compared to the Euclidean heuristic, which may underestimate the true distance due to diagonal movements not being allowed.")
print()

## runing more tests with different goals and initial positions
xI = (10,10)
xG = (30,16)
print("Testing different algorithms on medium maze. Initial state is (10,10) and goal state is (30,16).")
print()
action_list, cost_path, nodes_visited = depthFirstSearch(xI,xG,n,m,O)
print()
action_list, cost_path, nodes_visited = breadthFirstSearch(xI,xG,n,m,O)
print()
action_list, cost_path, nodes_visited = DijkstraSearch(xI,xG,n,m,O,cost=stayWestCost)
print()
action_list, cost_path, nodes_visited = DijkstraSearch(xI,xG,n,m,O,cost=stayEastCost)
print()
action_list, cost_path, nodes_visited = aStarSearch(xI,xG,n,m,O,heuristic = manhattanHeuristic)
print()
action_list, cost_path, nodes_visited = aStarSearch(xI,xG,n,m,O,heuristic = euclideanHeuristic)
print()
action_list, cost_path, nodes_visited = aStarSearch(xI,xG,n,m,O,heuristic = nullHeuristic)
print()

xI = (1,1)
xG = (18,11)
print("Testing different algorithms on medium maze. Initial state is (1,1) and goal state is (18,11).")
print()
action_list, cost_path, nodes_visited = depthFirstSearch(xI,xG,n,m,O)
print()
action_list, cost_path, nodes_visited = breadthFirstSearch(xI,xG,n,m,O)
print()
action_list, cost_path, nodes_visited = DijkstraSearch(xI,xG,n,m,O,cost=stayWestCost)
print()
action_list, cost_path, nodes_visited = DijkstraSearch(xI,xG,n,m,O,cost=stayEastCost)
print()
action_list, cost_path, nodes_visited = aStarSearch(xI,xG,n,m,O,heuristic = manhattanHeuristic)
print()
action_list, cost_path, nodes_visited = aStarSearch(xI,xG,n,m,O,heuristic = euclideanHeuristic)
print()
action_list, cost_path, nodes_visited = aStarSearch(xI,xG,n,m,O,heuristic = nullHeuristic)
print()

xI = (1,15)
xG = (1,1)
print("Testing different algorithms on medium maze. Initial state is (1,15) and goal state is (1,1).")
print()
action_list, cost_path, nodes_visited = depthFirstSearch(xI,xG,n,m,O)
print()
action_list, cost_path, nodes_visited = breadthFirstSearch(xI,xG,n,m,O)
print()
action_list, cost_path, nodes_visited = DijkstraSearch(xI,xG,n,m,O,cost=stayWestCost)
print()
action_list, cost_path, nodes_visited = DijkstraSearch(xI,xG,n,m,O,cost=stayEastCost)
print()
action_list, cost_path, nodes_visited = aStarSearch(xI,xG,n,m,O,heuristic = manhattanHeuristic)
print()
action_list, cost_path, nodes_visited = aStarSearch(xI,xG,n,m,O,heuristic = euclideanHeuristic)
print()
action_list, cost_path, nodes_visited = aStarSearch(xI,xG,n,m,O,heuristic = nullHeuristic)
print()



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
    
    
