# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    return successorGameState.getScore()

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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    """
    print "Depth:", self.depth 
    print "Actions:",gameState.getLegalActions()
    print "Number of Agents: ",gameState.getNumAgents()
    print "eval function: ",self.evaluationFunction(gameState)
    """
    
    # Generate legal moves for pacman
    legalMoves = gameState.getLegalActions()
    # Generate list of states from list of legal moves
    states = [gameState.generatePacmanSuccessor(action) for action in legalMoves]
    # Get the best scores from each state
    scores = [self.GetValue(state,0) for state in states]
    
    # Choose action with highest score
    bestScore = max(scores)
    print "*****Max score*******: ",bestScore
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]


  def GetValue(self,state,depth):
    """
    Returns the best score for an agent
    For pacman agent will return the maximum score
    For ghost agents will return the minimum score
    
    Routine is called recursively until max depth is reached or no further states can be generated
    """
    depth = depth + 1
    if self.TerminalTest(state,depth):
      return self.Utility(state)
    
    agentIndex = depth % state.getNumAgents()    
    legalMoves = state.getLegalActions(agentIndex)
    states = [state.generateSuccessor(agentIndex,action) for action in legalMoves]
    scores = [self.GetValue(state,depth) for state in states]

    if agentIndex == 0:
#      print "Return max,depth,scores: ",max(scores),depth,scores
      return max(scores)
    else:
#      print "Return min,depth,scores: ",min(scores),depth,scores
      return min(scores)
    
  def TerminalTest(self,state,depth):
    """
    Routine to determine if at end of recursion
    Returns true if recursion should end, false otherwise
    """
    legalMoves = state.getLegalActions()
    max_tree_depth = self.depth*state.getNumAgents()
    return (depth == max_tree_depth) or (len(legalMoves) == 0) 
    
  def Utility(self,state):
    """
    Routine to return score or value of a given state
    """
    return self.evaluationFunction(state)

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """
  def getAction(self, gameState):

    alpha = float("-inf")
    beta = float("inf")
    legalMoves = gameState.getLegalActions()
    states = [gameState.generatePacmanSuccessor(action) for action in legalMoves]
    scores = [self.GetValue(state,0,alpha,beta) for state in states]
    bestScore = max(scores)
    print "*****Max score*******: ",bestScore
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]


  def GetValue(self,state,depth,alpha,beta):
    """
    Returns the best score for an agent
    For pacman agent will return the maximum score
    For ghost agents will return the minimum score
    
    Routine is called recursively until max depth is reached or no further states can be generated
    Routine modified to pass in an alpha and beta value
    These values are kept track of at each depth in order to skip branches of the tree that cannot
    improve on current values
    """
    depth = depth + 1
    if self.TerminalTest(state,depth):
      return self.Utility(state)
    
    agentIndex = depth % state.getNumAgents()    
    legalMoves = state.getLegalActions(agentIndex)
    
# Pacman agent part
    if agentIndex == 0:
      v = float("-inf")
      
      for action in legalMoves:
        s = state.generateSuccessor(agentIndex,action)   
        v = max(v,self.GetValue(s,depth,alpha,beta))     
        if v >= beta:
          return v
        alpha = max(alpha,v)
      
      return v
# Ghost agents part
    else:
      v = float("inf")
      
      for action in legalMoves:
        s = state.generateSuccessor(agentIndex,action)   
        v = min(v,self.GetValue(s,depth,alpha,beta))     
        if v <= alpha:
          return v
        beta = min(beta,v)
      
      return v
        
    
  def TerminalTest(self,state,depth):
    agentIndex = depth % state.getNumAgents()    
    legalMoves = state.getLegalActions(agentIndex)
    max_tree_depth = self.depth*state.getNumAgents()
    return (depth == max_tree_depth) or (len(legalMoves) == 0) 
    
  def Utility(self,state):
    return self.evaluationFunction(state)

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
    legalMoves = gameState.getLegalActions()
    states = [gameState.generatePacmanSuccessor(action) for action in legalMoves]
    scores = [self.GetValue(state,0) for state in states]
    bestScore = max(scores)
    print "*****Max score*******: ",bestScore
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]


  def GetValue(self,state,depth):
    """
    Returns the best score for an agent
    For pacman agent will return the maximum score
    For ghost agents will return the minimum score
    
    Routine is called recursively until max depth is reached or no further states can be generated
    Routine modified to handle randomization of ghost.  For ghost agents average of scores is 
    return rather than minimum
    """
    depth = depth + 1
    if self.TerminalTest(state,depth):
      return self.Utility(state)
    
    agentIndex = depth % state.getNumAgents()    
    legalMoves = state.getLegalActions(agentIndex)
    states = [state.generateSuccessor(agentIndex,action) for action in legalMoves]
    scores = [self.GetValue(state,depth) for state in states]

    if agentIndex == 0:
#      print "Return max,depth,scores: ",max(scores),depth,scores
      return max(scores)
    else:
#     Return average of scores to account for randomization of ghost
      return sum(scores) / len(scores)
    
  def TerminalTest(self,state,depth):
    legalMoves = state.getLegalActions()
    max_tree_depth = self.depth*state.getNumAgents()
    return (depth == max_tree_depth) or (len(legalMoves) == 0) 
    
  def Utility(self,state):
    return self.evaluationFunction(state)

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

