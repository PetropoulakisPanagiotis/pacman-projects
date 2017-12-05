# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """

    from util import Stack

    # stackXY: ((x,y),[path]) #
    stackXY = Stack()

    visited = [] # Visited states
    path = [] # Every state keeps it's path from the starting state

    # Check if initial state is goal state #
    if problem.isGoalState(problem.getStartState()):
        return []

    # Start from the beginning and find a solution, path is an empty list #
    stackXY.push((problem.getStartState(),[]))

    while(True):

        # Terminate condition: can't find solution #
        if stackXY.isEmpty():
            return []

        # Get informations of current state #
        xy,path = stackXY.pop() # Take position and path
        visited.append(xy)

        # Comment this and uncomment 125. This only works for autograder    #
        # In lectures we check if a state is a goal when we find successors #

        # Terminate condition: reach goal #
        if problem.isGoalState(xy):
            return path

        # Get successors of current state #
        succ = problem.getSuccessors(xy)

        # Add new states in stack and fix their path #
        if succ:
            for item in succ:
                if item[0] not in visited:

                # Lectures code:
                # All impementations run in autograder and in comments i write
                # the proper code that i have been taught in lectures
                # if item[0] not in visited and item[0] not in (state[0] for state in stackXY.list):
                #   if problem.isGoalState(item[0]):
                #       return path + [item[1]]

                    newPath = path + [item[1]] # Calculate new path
                    stackXY.push((item[0],newPath))

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    from util import Queue

    # queueXY: ((x,y),[path]) #
    queueXY = Queue()

    visited = [] # Visited states
    path = [] # Every state keeps it's path from the starting state

    # Check if initial state is goal state #
    if problem.isGoalState(problem.getStartState()):
        return []

    # Start from the beginning and find a solution, path is empty list #
    queueXY.push((problem.getStartState(),[]))

    while(True):

        # Terminate condition: can't find solution #
        if queueXY.isEmpty():
            return []

        # Get informations of current state #
        xy,path = queueXY.pop() # Take position and path
        visited.append(xy)

        # Comment this and uncomment 179. This is only works for autograder
        # In lectures we check if a state is a goal when we find successors

        # Terminate condition: reach goal #
        if problem.isGoalState(xy):
            return path

        # Get successors of current state #
        succ = problem.getSuccessors(xy)

        # Add new states in queue and fix their path #
        if succ:
            for item in succ:
                if item[0] not in visited and item[0] not in (state[0] for state in queueXY.list):

                    # Lectures code:
                    # All impementations run in autograder and in comments i write
                    # the proper code that i have been taught in lectures
                    # if problem.isGoalState(item[0]):
                    #   return path + [item[1]]

                    newPath = path + [item[1]] # Calculate new path
                    queueXY.push((item[0],newPath))

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    from util import PriorityQueue

    # queueXY: ((x,y),[path],priority) #
    queueXY = PriorityQueue()

    visited = [] # Visited states
    path = [] # Every state keeps it's path from the starting state

    # Check if initial state is goal state #
    if problem.isGoalState(problem.getStartState()):
        return []

    # Start from the beginning and find a solution, path is empty list #
    # with the cheapest priority                                       #
    queueXY.push((problem.getStartState(),[]),0)

    while(True):

        # Terminate condition: can't find solution #
        if queueXY.isEmpty():
            return []

        # Get informations of current state #
        xy,path = queueXY.pop() # Take position and path
        visited.append(xy)

        # This only works for autograder    #
        # In lectures we check if a state is a goal when we find successors #

        # Terminate condition: reach goal #
        if problem.isGoalState(xy):
            return path

        # Get successors of current state #
        succ = problem.getSuccessors(xy)

        # Add new states in queue and fix their path #
        if succ:
            for item in succ:
                if item[0] not in visited and (item[0] not in (state[2][0] for state in queueXY.heap)):

                    #    Like previous algorithms: we should check in this point if successor
                    #    is a goal state so as to follow lectures code

                    newPath = path + [item[1]]
                    pri = problem.getCostOfActions(newPath)

                    queueXY.push((item[0],newPath),pri)

                # State is in queue. Check if current path is cheaper from the previous one #
                elif item[0] not in visited and (item[0] in (state[2][0] for state in queueXY.heap)):
                    for state in queueXY.heap:
                        if state[2][0] == item[0]:
                            oldPri = problem.getCostOfActions(state[2][1])

                    newPri = problem.getCostOfActions(path + [item[1]])

                    # State is cheaper with his hew father -> update and fix parent #
                    if oldPri > newPri:
                        newPath = path + [item[1]]
                        queueXY.update((item[0],newPath),newPri)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

from util import PriorityQueue
class MyPriorityQueueWithFunction(PriorityQueue):
    """
    Implements a priority queue with the same push/pop signature of the
    Queue and the Stack classes. This is designed for drop-in replacement for
    those two classes. The caller has to provide a priority function, which
    extracts each item's priority.
    """
    def  __init__(self, problem, priorityFunction):
        "priorityFunction (item) -> priority"
        self.priorityFunction = priorityFunction      # store the priority function
        PriorityQueue.__init__(self)        # super-class initializer
        self.problem = problem
    def push(self, item, heuristic):
        "Adds an item to the queue with priority from the priority function"
        PriorityQueue.push(self, item, self.priorityFunction(self.problem,item,heuristic))

# Calculate f(n) = g(n) + h(n) #
def f(problem,state,heuristic):

    return problem.getCostOfActions(state[1]) + heuristic(state[0],problem)

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # queueXY: ((x,y),[path]) #
    queueXY = MyPriorityQueueWithFunction(problem,f)

    path = [] # Every state keeps it's path from the starting state
    visited = [] # Visited states


    # Check if initial state is goal state #
    if problem.isGoalState(problem.getStartState()):
        return []

    # Add initial state. Path is an empty list #
    element = (problem.getStartState(),[])

    queueXY.push(element,heuristic)

    while(True):

        # Terminate condition: can't find solution #
        if queueXY.isEmpty():
            return []

        # Get informations of current state #
        xy,path = queueXY.pop() # Take position and path

        # State is already been visited. A path with lower cost has previously
        # been found. Overpass this state
        if xy in visited:
            continue

        visited.append(xy)

        # Terminate condition: reach goal #
        if problem.isGoalState(xy):
            return path

        # Get successors of current state #
        succ = problem.getSuccessors(xy)

        # Add new states in queue and fix their path #
        if succ:
            for item in succ:
                if item[0] not in visited:

                    # Like previous algorithms: we should check in this point if successor
                    # is a goal state so as to follow lectures code

                    newPath = path + [item[1]] # Fix new path
                    element = (item[0],newPath)
                    queueXY.push(element,heuristic)

# Editor:
# Sdi1500129
# Petropoulakis Panagiotis

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
