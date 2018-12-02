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

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    # followind slides anazitisi_se_grafous.pdf page 11 but implemented with Stack

    # initialize the frontier using the initial state of problem
    frontier = util.Stack()
    # the stack will have tuples that contain the state of the node and the actions to this node
    frontier.push((problem.getStartState(), []))

    # initialize the explored list to be empty
    explored = []

    while (True):
        # if the frontier is empty then return failure
        if frontier.isEmpty():
            return False

        # choose a leaf node and remove it from the frontier
        poppedNode = frontier.pop()
        state = poppedNode[0]
        actions = poppedNode[1]

        # if the node contains a goal state then return the corresponding solution
        if problem.isGoalState(state):
            return actions    #return actions to goal state

        # add the state of the node to the explored set
        explored.append(state)

        # expand the popped node
        successors = problem.getSuccessors(state)

        # add the resulting nodes to the frontier and their paths
        # only if their state is not in the frontier or the explored set
        for s in successors:
            if s[0] not in explored and s[0] not in (state[0] for state in frontier.list):
                frontier.push((s[0], actions + [s[1]]))


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    # followind slides anazitisi_se_grafous.pdf page 27

    # if initial state is goal state return empty actions
    if problem.isGoalState(problem.getStartState()):
        return []

    # frontier <- a FIFO queue with node as the only element
    frontier = util.Queue()
    # the queue will have tuples that contain the state of the node and the actions to this node
    frontier.push((problem.getStartState(), []))

    # explored <- an empty list
    explored = []

    while (True):
        # if the frontier is empty then return failure
        if frontier.isEmpty():
            return False

        # node <- POP(frontier)
        poppedNode = frontier.pop()
        state = poppedNode[0]
        actions = poppedNode[1]

        # add the state of the node to the explored set
        explored.append(state)
        # expand the popped node
        successors = problem.getSuccessors(state)
        # for each action in problem.ACTIONS(node.STATE) do
        for s in successors:
            # if child.STATE is not in explored or frontier then
            if s[0] not in explored and s[0] not in (state[0] for state in frontier.list):
                # if problem.GOAL-TEST(child.STATE) then return SOLUTION(child)
                if problem.isGoalState(s[0]):
                    return actions + [s[1]]
                # frontier <- INSERT(child, frontier)
                frontier.push((s[0], actions + [s[1]]))



def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    # followind slides anazitisi_se_grafous.pdf page 35

    # frontier <- a priority queue ordered by PATH-COST, with node as the only element
    frontier = util.PriorityQueue()
    # the queue will have tuples that contain the state of the node and the actions to this node
    frontier.push((problem.getStartState(), []), 0)

    # explored <- an empty list
    explored = []

    while (True):
        # if the frontier is empty then return failure
        if frontier.isEmpty():
            return False

        # node <- POP(frontier)
        poppedNode = frontier.pop()
        state = poppedNode[0]
        actions = poppedNode[1]

        # if node state is goal state return actions
        if problem.isGoalState(state):
            return actions

        # add the state of the node to the explored list
        explored.append(state)

        # expand the popped node
        successors = problem.getSuccessors(state)

        # for each action in problem.ACTIONS(node.STATE) do
        for s in successors:
            # if child.STATE is not in explored or frontier then
            if s[0] not in explored and s[0] not in (state[2][0] for state in frontier.heap):
                # frontier <- INSERT(child, frontier)
                cost = problem.getCostOfActions(actions + [s[1]])
                frontier.push((s[0], actions + [s[1]]), cost)
            else:
                for state in frontier.heap:
                    # if child.STATE is in frontier with higher PATH-COST
                    if state[2][0] == s[0]:
                        frontierCost = problem.getCostOfActions(state[2][1])
                        newCost = problem.getCostOfActions(actions + [s[1]])

                        if frontierCost > newCost:
                            # replace that frontier node with child
                            frontier.update((s[0], actions + [s[1]]), newCost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    # followind slides anazitisi_se_grafous.pdf page 35 but with combination of cost and heuristic

    # frontier <- a priority queue ordered by PATH-COST, with node as the only element
    frontier = util.PriorityQueue()
    # the queue will have tuples that contain the state of the node and the actions to this node
    frontier.push((problem.getStartState(), []), 0)

    # explored <- an empty list
    explored = []

    while (True):
        # if the frontier is empty then return failure
        if frontier.isEmpty():
            return False

        poppedNode = frontier.pop()
        state = poppedNode[0]
        actions = poppedNode[1]

        # if node state is goal state return actions
        if problem.isGoalState(state):
            return actions

        # add the state of the node to the explored set
        explored.append(state)

        # expand the popped node
        successors = problem.getSuccessors(state)

        # for each action in problem.ACTIONS(node.STATE) do
        for s in successors:
            # if child.STATE is not in explored or frontier then
            if s[0] not in explored and s[0] not in (state[2][0] for state in frontier.heap):
                # frontier <- INSERT(child, frontier)
                cost = problem.getCostOfActions(actions + [s[1]])
                frontier.push((s[0], actions + [s[1]]), cost + heuristic(s[0], problem))
            else:
                for state in frontier.heap:
                    # if child.STATE is in frontier with higher PATH-COST + heuristic
                    if state[2][0] == s[0]:
                        frontierCost = problem.getCostOfActions(state[2][1])
                        newCost = problem.getCostOfActions(actions + [s[1]])

                        if frontierCost + heuristic(s[0], problem) > newCost + heuristic(s[0], problem):
                            # replace that frontier node with child
                            frontier.update((s[0], actions + [s[1]]), newCost + heuristic(s[0], problem))


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
