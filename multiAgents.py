# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        from util import manhattanDistance

        food = newFood.asList()
        foodDistances = []
        ghostDistances = []
        count = 0 # Tell as how is it worthing to pick this state

        # Calculate distance from every food #
        for item in food:
            foodDistances.append(manhattanDistance(newPos,item))

        # Fix count based on food     #
        # Close food -> better state  #
        for i in foodDistances:
            if i <= 4:
                count += 1
            elif i > 4 and i <= 15:
                count += 0.2
            else:
                count += 0.15

        # Calculate distance from every ghost #
        for ghost in successorGameState.getGhostPositions():
            ghostDistances.append(manhattanDistance(ghost,newPos))

        # Fix coun based on ghosts   #
        # Close ghost -> worst state #
        for ghost in successorGameState.getGhostPositions():
            if ghost == newPos: # We reached a ghost
                count = 2 - count

            elif manhattanDistance(ghost,newPos) <= 3.5:
                count = 1 - count

        return successorGameState.getScore() + count

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        """
            miniMax: receives state, agent(0,1,2...) and current depth
            miniMax: returns a list: [cost,action]
            Example with depth: 3
            That means pacman played 3 times and all ghosts played 3 times
        """

        def miniMax(gameState,agent,depth):
            result = []

            # Terminate state #
            if not gameState.getLegalActions(agent):
                return self.evaluationFunction(gameState),0

            # Reached max depth #
            if depth == self.depth:
                return self.evaluationFunction(gameState),0

            # All ghosts have finised one round: increase depth(last ghost) #
            if agent == gameState.getNumAgents() - 1:
                depth += 1

            # Calculate nextAgent #

            # Last ghost: nextAgent = pacman #
            if agent == gameState.getNumAgents() - 1:
                nextAgent = self.index

            # Availiable ghosts. Pick next ghost #
            else:
                nextAgent = agent + 1

            # For every successor find minimax value #
            for action in gameState.getLegalActions(agent):

                if not result: # First move
                    nextValue = miniMax(gameState.generateSuccessor(agent,action),nextAgent,depth)

                    # Fix result with minimax value and action #
                    result.append(nextValue[0])
                    result.append(action)
                else:

                    # Check if miniMax value is better than the previous one #

                    previousValue = result[0] # Keep previous value. Minimax
                    nextValue = miniMax(gameState.generateSuccessor(agent,action),nextAgent,depth)

                    # Max agent: Pacman #
                    if agent == self.index:
                        if nextValue[0] > previousValue:
                            result[0] = nextValue[0]
                            result[1] = action

                    # Min agent: Ghost #
                    else:
                        if nextValue[0] < previousValue:
                            result[0] = nextValue[0]
                            result[1] = action
            return result

        # Call minMax with initial depth = 0 and get an action #
        # Pacman plays first -> agent == 0 or self.index       #

        return miniMax(gameState,self.index,0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agentwith alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        """
            AB: receives a state an agent(0,1,2...) and current depth
            AB: return a list:[cost,action]
            Example with depth: 3
            That means pacman played 3 times and all ghosts 3 times
            AB: Cuts some nodes. That means we can use higher depth in the same
            time of minimax algorithm in lower depth
        """

        def AB(gameState,agent,depth,a,b):
            result = []

            # Terminate state #
            if not gameState.getLegalActions(agent):
                return self.evaluationFunction(gameState),0

            # Reached max depth #
            if depth == self.depth:
                return self.evaluationFunction(gameState),0

            # All ghosts have finised one round: increase depth #
            if agent == gameState.getNumAgents() - 1:
                depth += 1

            # Calculate nextAgent #

            # Last ghost: nextAgent = pacman #
            if agent == gameState.getNumAgents() - 1:
                nextAgent = self.index

            # Availiable ghosts. Pick next ghost #
            else:
                nextAgent = agent + 1

            # For every successor find minmax value #
            for action in gameState.getLegalActions(agent):
                if not result: # First move
                    nextValue = AB(gameState.generateSuccessor(agent,action),nextAgent,depth,a,b)

                    # Fix result #
                    result.append(nextValue[0])
                    result.append(action)

                    # Fix initial a,b (for the first node) #
                    if agent == self.index:
                        a = max(result[0],a)
                    else:
                        b = min(result[0],b)
                else:
                    # Check if minMax value is better than the previous one #
                    # Chech if we can overpass some nodes                   #

                    # There is no need to search next nodes                 #
                    # AB Prunning is true                                   #
                    if result[0] > b and agent == self.index:
                        return result

                    if result[0] < a and agent != self.index:
                        return result

                    previousValue = result[0] # Keep previous value
                    nextValue = AB(gameState.generateSuccessor(agent,action),nextAgent,depth,a,b)

                    # Max agent: Pacman #
                    if agent == self.index:
                        if nextValue[0] > previousValue:
                            result[0] = nextValue[0]
                            result[1] = action
                            # a may change #
                            a = max(result[0],a)

                    # Min agent: Ghost #
                    else:
                        if nextValue[0] < previousValue:
                            result[0] = nextValue[0]
                            result[1] = action
                            # b may change #
                            b = min(result[0],b)
            return result

        # Call AB with initial depth = 0 and -inf and inf(a,b) values      #
        # Get an action                                                    #
        # Pacman plays first -> self.index                                 #
        return AB(gameState,self.index,0,-float("inf"),float("inf"))[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        """
        expectiMax: receives state, agent(0,1,2...) and current depth
        expectiMax: returns a list: [cost,action]
        Example with depth: 3
        That means pacman played 3 times and all ghosts played 3 times
        Now ghosts move randomly. The difference with minimax is that
        ghosts may not play the best move. We can win more easily.
        This is close to our world, as no one playes optimal every time
        We can imagine chance nodes like min nodes. But their value is equal
        with the sum: (moveValue * probability) for every move
        """

        def expectiMax(gameState,agent,depth):
            result = []

            # Terminate state #
            if not gameState.getLegalActions(agent):
                return self.evaluationFunction(gameState),0

            # Reached max depth #
            if depth == self.depth:
                return self.evaluationFunction(gameState),0

            # All ghosts have finised one round: increase depth(last ghost) #
            if agent == gameState.getNumAgents() - 1:
                depth += 1

            # Calculate nextAgent #

            # Last ghost: nextAgent = pacman #
            if agent == gameState.getNumAgents() - 1:
                nextAgent = self.index

            # Availiable ghosts. Pick next ghost #
            else:
                nextAgent = agent + 1

            # For every successor find minimax value #
            for action in gameState.getLegalActions(agent):
                if not result: # First move
                    nextValue = expectiMax(gameState.generateSuccessor(agent,action),nextAgent,depth)
                    # Fix chance node                               #
                    # Probability: 1 / p -> 1 / total legal actions #
                    # Ghost pick an action based in 1 / p. As all   #
                    # actions have the same probability             #
                    if(agent != self.index):
                        result.append((1.0 / len(gameState.getLegalActions(agent))) * nextValue[0])
                        result.append(action)
                    else:
                        # Fix result with minimax value and action #
                        result.append(nextValue[0])
                        result.append(action)
                else:

                    # Check if miniMax value is better than the previous one #
                    previousValue = result[0] # Keep previous value. Minimax
                    nextValue = expectiMax(gameState.generateSuccessor(agent,action),nextAgent,depth)

                    # Max agent: Pacman #
                    if agent == self.index:
                        if nextValue[0] > previousValue:
                            result[0] = nextValue[0]
                            result[1] = action

                    # Min agent: Ghost                                         #
                    # Now we don't select a better action but we continue to   #
                    # calculate our sum to find the total value of chance node #
                    else:
                        result[0] = result[0] + (1.0 / len(gameState.getLegalActions(agent))) * nextValue[0]
                        result[1] = action
            return result

        # Call expectiMax with initial depth = 0 and get an action  #
        # Pacman plays first -> agent == 0 or self.index            #
        # We can will more likely than minimax. Ghosts may not play #
        # optimal in some cases                                     #

        return expectiMax(gameState,self.index,0)[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    pacmanPosition = currentGameState.getPacmanPosition()
    activeGhosts = [] # Keep active ghosts(can eat pacman)
    scaredGhosts = [] # Keep scared ghosts(pacman should eat them for extra points)
    totalCapsules = len(currentGameState.getCapsules()) # Keep total capsules
    totalFood = len(food) # Keep total remaining food
    myEval = 0 # Evaluation value

    # Fix active and scared ghosts #
    for ghost in ghosts:
        if ghost.scaredTimer: # Is scared ghost
            scaredGhosts.append(ghost)
        else:
            activeGhosts.append(ghost)

    # Score weight: 1.5                                    #
    # Better score -> better result                        #
    # Score tell us some informations about current state  #
    # but the weight is low. We don't care enough for this #
    # But we do not want to lose                           #
    myEval += 1.5 * currentGameState.getScore()

    # Food weight: -10                                   #
    # Pacman will receive 10 points if he eats one food  #
    # If our state has a lot of food this is very bad    #
    # We are far away from our goal. If pacman eats food #
    # Evaluation value will be better in the new         #
    # state, because remaining food is less              #
    myEval += -10 * totalFood

    # Capsules weight: -20                            #
    # Same like food but pacman gains a huge amount   #
    # of points if he eats ghosts. So our goal is to  #
    # eat a capsule and then eat a ghost              #
    # For that reason pacman should eat capsules more #
    # frequently than food                            #
    # Weight food < Weight capsules                   #
    myEval += -20 * totalCapsules

    # Keep distances from food, active and scared ghosts #
    foodDistances = []
    activeGhostsDistances = []
    scaredGhostsDistances = []

    # Find distances #
    for item in food:
        foodDistances.append(manhattanDistance(pacmanPosition,item))

    for item in activeGhosts:
        scaredGhostsDistances.append(manhattanDistance(pacmanPosition,item.getPosition()))

    for item in scaredGhosts:
        scaredGhostsDistances.append(manhattanDistance(pacmanPosition,item.getPosition()))

    # Fix evaluation based on food distances  #
    # It is very bad for pacman to have close #
    # food. He must eat it.                   #
    # Close food weight: -1                   #
    # Quite close food weight: -0.5           #
    # Far away food weight: -0.2              #
    for item in foodDistances:
        if item < 3:
            myEval += -1 * item
        if item < 7:
            myEval += -0.5 * item
        else:
            myEval += -0.2 * item

    # Fix evaluation based on scared ghosts distances    #
    # It is very bad for pacman to have close scared     #
    # ghosts. He must eat them so as to gain many points #
    # We should prefer to eat a ghost rather than eat a  #
    # close food                                         #
    # Close scared ghosts weight: -20                    #
    # Quite close scared ghosts weight: -10              #
    for item in scaredGhostsDistances:
        if item < 3:
            myEval += -20 * item
        else:
            myEval += -10 * item

    # Fix evaluation base on active ghosts distances    #
    # Pacman should avoid active ghosts                 #
    # Close ghost weight: 3                             #
    # Quite close ghost weight: 2                       #
    # Far away ghosts weight: 0.5                       #
    # We should prefer ghosts remaining far away        #
    for item in activeGhostsDistances:
        if item < 3:
            myEval += 3 * item
        elif item < 7:
            myEval += 2 * item
        else:
            myEval += 0.5 * item

    return myEval

# Abbreviation
better = betterEvaluationFunction

# Editor:
# sdi1500129
# Petropoulakis Panagiotis