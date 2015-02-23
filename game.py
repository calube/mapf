# game.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

#import util
from util import *
from util import raiseNotDefined
import time, os
import traceback
import itertools 
from itertools import cycle

try:
  import boinc
  _BOINC_ENABLED = True
except:
  _BOINC_ENABLED = False

#######################
# Parts worth reading #
#######################

class ConstraintTree:
    def __init__(self,rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None

    def insertLeft(self,newNode):
        if self.leftChild == None:
            self.leftChild = ConstraintTree(newNode)
        else:
            t = ConstraintTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t

    def insertRight(self,newNode):
        if self.rightChild == None:
            self.rightChild = ConstraintTree(newNode)
        else:
            t = ConstraintTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t


    def getRightChild(self):
        return self.rightChild

    def getLeftChild(self):
        return self.leftChild

    def setRootVal(self,obj):
        self.key = obj

    def getRootVal(self):
        return self.key

class Agent:
  """
  An agent must define a getAction method, but may also define the
  following methods which will be called if they exist:

  def registerInitialState(self, state): # inspects the starting state
  """
  def __init__(self, index=0):
    self.index = index

  def getAction(self, state):
    """
    The Agent will receive a GameState (from either {pacman, capture, sonar}.py) and
    must return an action from Directions.{North, South, East, West, Stop}
    """
    raiseNotDefined()

class Directions:
  NORTH = 'North'
  SOUTH = 'South'
  EAST = 'East'
  WEST = 'West'
  STOP = 'Stop'

  NUM = {
  NORTH: 1,
  SOUTH: 2,
  EAST: 3, 
  WEST: 4,
  STOP: 0}

  LEFT =       {NORTH: WEST,
                 SOUTH: EAST,
                 EAST:  NORTH,
                 WEST:  SOUTH,
                 STOP:  STOP}

  RIGHT =      dict([(y,x) for x, y in LEFT.items()])

  REVERSE = {NORTH: SOUTH,
             SOUTH: NORTH,
             EAST: WEST,
             WEST: EAST,
             STOP: STOP}

class Configuration:
  """
  A Configuration holds the (x,y) coordinate of a character, along with its
  traveling direction.

  The convention for positions, like a graph, is that (0,0) is the lower left corner, x increases
  horizontally and y increases vertically.  Therefore, north is the direction of increasing y, or (0,1).
  """

  def __init__(self, pos, direction):
    self.pos = pos
    self.direction = direction

  def getPosition(self):
    return (self.pos)

  def getDirection(self):
    return self.direction

  def isInteger(self):
    x,y = self.pos
    return x == int(x) and y == int(y)

  def __eq__(self, other):
    if other == None: return False
    return (self.pos == other.pos and self.direction == other.direction)

  def __hash__(self):
    x = hash(self.pos)
    y = hash(self.direction)
    return hash(x + 13 * y)

  def __str__(self):
    return "(x,y)="+str(self.pos)+", "+str(self.direction)

  def generateSuccessor(self, vector):
    """
    Generates a new configuration reached by translating the current
    configuration by the action vector.  This is a low-level call and does
    not attempt to respect the legality of the movement.

    Actions are movement vectors.
    """
    x, y= self.pos
    dx, dy = vector
    #print "vector: *** " , vector
    direction = Actions.vectorToDirection(vector)
    if direction == Directions.STOP:
      direction = self.direction # There is no stop direction
    return Configuration((x + dx, y+dy), direction)

class AgentState:
  """
  AgentStates hold the state of an agent (configuration, speed, scared, etc).
  """

  def __init__( self, startConfiguration, isPacman ):
    self.start = startConfiguration
    self.configuration = startConfiguration
    self.isPacman = isPacman
    self.scaredTimer = 0

  def __str__( self ):
    if self.isPacman:
      return "Pacman: " + str( self.configuration )
    else:
      return "Ghost: " + str( self.configuration )

  def __eq__( self, other ):
    if other == None:
      return False
    return self.configuration == other.configuration and self.scaredTimer == other.scaredTimer

  def __hash__(self):
    return hash(hash(self.configuration) + 13 * hash(self.scaredTimer))

  def copy( self ):
    state = AgentState( self.start, self.isPacman )
    state.configuration = self.configuration
    state.scaredTimer = self.scaredTimer
    return state

  def getPosition(self):
    if self.configuration == None: return None
    return self.configuration.getPosition()

  def getDirection(self):
    return self.configuration.getDirection()

class Grid:
  """
  A 2-dimensional array of objects backed by a list of lists.  Data is accessed
  via grid[x][y] where (x,y) are positions on a Pacman map with x horizontal,
  y vertical and the origin (0,0) in the bottom left corner.

  The __str__ method constructs an output that is oriented like a pacman board.
  """
  def __init__(self, width, height, initialValue=False, bitRepresentation=None):
    if initialValue not in [False, True]: raise Exception('Grids can only contain booleans')
    self.CELLS_PER_INT = 30

    self.width = width
    self.height = height
    self.data = [[initialValue for y in range(height)] for x in range(width)]
    if bitRepresentation:
      self._unpackBits(bitRepresentation)

  def __getitem__(self, i):
    return self.data[i]

  def __setitem__(self, key, item):
    self.data[key] = item

  def __str__(self):
    out = [[str(self.data[x][y])[0] for x in range(self.width)] for y in range(self.height)]
    out.reverse()
    return '\n'.join([''.join(x) for x in out])

  def __eq__(self, other):
    if other == None: return False
    return self.data == other.data

  def __hash__(self):
    # return hash(str(self))
    base = 1
    h = 0
    for l in self.data:
      for i in l:
        if i:
          h += base
        base *= 2
    return hash(h)

  def copy(self):
    g = Grid(self.width, self.height)
    g.data = [x[:] for x in self.data]
    return g

  def deepCopy(self):
    return self.copy()

  def shallowCopy(self):
    g = Grid(self.width, self.height)
    g.data = self.data
    return g

  def count(self, item =True ):
    return sum([x.count(item) for x in self.data])

  def asList(self, key = True):
    list = []
    for x in range(self.width):
      for y in range(self.height):
        if self[x][y] == key: list.append( (x,y) )
    return list

  def packBits(self):
    """
    Returns an efficient int list representation

    (width, height, bitPackedInts...)
    """
    bits = [self.width, self.height]
    currentInt = 0
    for i in range(self.height * self.width):
      bit = self.CELLS_PER_INT - (i % self.CELLS_PER_INT) - 1
      x, y = self._cellIndexToPosition(i)
      if self[x][y]:
        currentInt += 2 ** bit
      if (i + 1) % self.CELLS_PER_INT == 0:
        bits.append(currentInt)
        currentInt = 0
    bits.append(currentInt)
    return tuple(bits)

  def _cellIndexToPosition(self, index):
    x = index / self.height
    y = index % self.height
    return x, y

  def _unpackBits(self, bits):
    """
    Fills in data from a bit-level representation
    """
    cell = 0
    for packed in bits:
      for bit in self._unpackInt(packed, self.CELLS_PER_INT):
        if cell == self.width * self.height: break
        x, y = self._cellIndexToPosition(cell)
        self[x][y] = bit
        cell += 1

  def _unpackInt(self, packed, size):
    bools = []
    if packed < 0: raise ValueError, "must be a positive integer"
    for i in range(size):
      n = 2 ** (self.CELLS_PER_INT - i - 1)
      if packed >= n:
        bools.append(True)
        packed -= n
      else:
        bools.append(False)
    return bools

def reconstituteGrid(bitRep):
  if type(bitRep) is not type((1,2)):
    return bitRep
  width, height = bitRep[:2]
  return Grid(width, height, bitRepresentation= bitRep[2:])

####################################
# Parts you shouldn't have to read #
####################################

class Actions:
  """
  A collection of static methods for manipulating move actions.
  """
  # Directions
  _directions = {Directions.NORTH: (0, 1),
                 Directions.SOUTH: (0, -1),
                 Directions.EAST:  (1, 0),
                 Directions.WEST:  (-1, 0),
                 Directions.STOP:  (0, 0)}           

  _directionsAsList = _directions.items()

  TOLERANCE = .001

  def reverseDirection(action):
    if action == Directions.NORTH:
      return Directions.SOUTH
    if action == Directions.SOUTH:
      return Directions.NORTH
    if action == Directions.EAST:
      return Directions.WEST
    if action == Directions.WEST:
      return Directions.EAST
    return action
  reverseDirection = staticmethod(reverseDirection)

  def vectorToDirection(vector):
    dx, dy = vector
    if dy > 0:
      return Directions.NORTH
    if dy < 0:
      return Directions.SOUTH
    if dx < 0:
      return Directions.WEST
    if dx > 0:
      return Directions.EAST
    return Directions.STOP
  vectorToDirection = staticmethod(vectorToDirection)

  def directionToVector(direction, speed = 1.0):
    dx, dy =  Actions._directions[direction]
    return (dx * speed, dy * speed)
  directionToVector = staticmethod(directionToVector)

  def getPossibleActions(config, walls):
    possible = []
    x, y = config.pos
    x_int, y_int = int(x + 0.5), int(y + 0.5)

    # In between grid points, all agents must continue straight
    if (abs(x - x_int) + abs(y - y_int)  > Actions.TOLERANCE):
      return [config.getDirection()]

    for dir, vec in Actions._directionsAsList:
      dx, dy = vec
      next_y = y_int + dy
      next_x = x_int + dx
      if not walls[next_x][next_y]: possible.append(dir)

    return possible

  getPossibleActions = staticmethod(getPossibleActions)

  def getLegalNeighbors(position, walls):
    x,y = position
    x_int, y_int = int(x + 0.5), int(y + 0.5)
    neighbors = []
    for dir, vec in Actions._directionsAsList:
      dx, dy = vec
      next_x = x_int + dx
      if next_x < 0 or next_x == walls.width: continue
      next_y = y_int + dy
      if next_y < 0 or next_y == walls.height: continue
      if not walls[next_x][next_y]: neighbors.append((next_x, next_y))
    return neighbors
  getLegalNeighbors = staticmethod(getLegalNeighbors)

  def getSuccessor(position, action):
    dx, dy = Actions.directionToVector(action)
    print "\t \t \t THIS IS ACTION : ", action
    x, y = position
    return (x + dx, y + dy)
  getSuccessor = staticmethod(getSuccessor)

class GameStateData:
  """

  """
  def __init__( self, prevState = None ):
    """
    Generates a new data packet by copying information from its predecessor.
    """
    if prevState != None:
      self.food = prevState.food.shallowCopy()
      self.capsules = prevState.capsules[:]
      self.agentStates = self.copyAgentStates( prevState.agentStates )
      self.layout = prevState.layout
      self._eaten = prevState._eaten
      self.score = prevState.score
    self._foodEaten = None
    self._capsuleEaten = None
    self._agentMoved = None
    self._lose = False
    self._win = False
    self.scoreChange = 0

  def deepCopy( self ):
    state = GameStateData( self )
    state.food = self.food.deepCopy()
    state.layout = self.layout.deepCopy()
    state._agentMoved = self._agentMoved
    state._foodEaten = self._foodEaten
    state._capsuleEaten = self._capsuleEaten
    return state

  def copyAgentStates( self, agentStates ):
    copiedStates = []
    for agentState in agentStates:
      copiedStates.append( agentState.copy() )
    return copiedStates

  def __eq__( self, other ):
    """
    Allows two states to be compared.
    """
    if other == None: return False
    # TODO Check for type of other
    if not self.agentStates == other.agentStates: return False
    if not self.food == other.food: return False
    if not self.capsules == other.capsules: return False
    if not self.score == other.score: return False
    return True

  def __hash__( self ):
    """
    Allows states to be keys of dictionaries.
    """
    for i, state in enumerate( self.agentStates ):
      try:
        int(hash(state))
      except TypeError, e:
        print e
        #hash(state)
    return int((hash(tuple(self.agentStates)) + 13*hash(self.food) + 113* hash(tuple(self.capsules)) + 7 * hash(self.score)) % 1048575 )

  def __str__( self ):
    width, height = self.layout.width, self.layout.height
    map = Grid(width, height)
    if type(self.food) == type((1,2)):
      self.food = reconstituteGrid(self.food)
    for x in range(width):
      for y in range(height):
        food, walls = self.food, self.layout.walls
        map[x][y] = self._foodWallStr(food[x][y], walls[x][y])

    for agentState in self.agentStates:
      if agentState == None: continue
      if agentState.configuration == None: continue
      x,y = [int( i ) for i in nearestPoint( agentState.configuration.pos )]
      agent_dir = agentState.configuration.direction
      if agentState.isPacman:
        map[x][y] = self._pacStr( agent_dir )
      else:
        map[x][y] = self._ghostStr( agent_dir )

    for x, y in self.capsules:
      map[x][y] = 'o'

    return str(map) + ("\nScore: %d\n" % self.score)

  def _foodWallStr( self, hasFood, hasWall ):
    if hasFood:
      return '.'
    elif hasWall:
      return '%'
    else:
      return ' '

  def _pacStr( self, dir ):
    if dir == Directions.NORTH:
      return 'v'
    if dir == Directions.SOUTH:
      return '^'
    if dir == Directions.WEST:
      return '>'
    return '<'

  def _ghostStr( self, dir ):
    return 'G'
    if dir == Directions.NORTH:
      return 'M'
    if dir == Directions.SOUTH:
      return 'W'
    if dir == Directions.WEST:
      return '3'
    return 'E'

  def initialize( self, layout, numGhostAgents ):
    """
    Creates an initial game state from a layout array (see layout.py).
    """
    self.food = layout.food.copy()
    self.capsules = layout.capsules[:]
    self.layout = layout
    self.score = 0
    self.scoreChange = 0

    self.agentStates = []
    numGhosts = 0
    for isPacman, pos in layout.agentPositions:
      if not isPacman:
        if numGhosts == numGhostAgents: continue # Max ghosts reached already
        else: numGhosts += 1
      self.agentStates.append( AgentState( Configuration( pos, Directions.STOP), isPacman) )
    self._eaten = [False for a in self.agentStates]

class Game:
  """
  The Game manages the control flow, soliciting actions from agents.
  """

  def __init__( self, agents, display, rules, startingIndex=0, muteAgents=False, catchExceptions=False, costFn = lambda x: 1 ):
    self.agentCrashed = False
    self.agents = agents
    self.display = display
    self.rules = rules
    self.startingIndex = startingIndex
    self.gameOver = False
    self.muteAgents = muteAgents
    self.catchExceptions = catchExceptions
    self.moveHistory = []
    self.totalAgentTimes = [0 for agent in agents]
    self.totalAgentTimeWarnings = [0 for agent in agents]
    self.agentTimeout = False
    self.costFn = costFn
    self._expanded = 0
    self._visited = {}
    self._visitedlist = []

  def getProgress(self):
    if self.gameOver:
      return 1.0
    else:
      return self.rules.getProgress(self)

  def _agentCrash( self, agentIndex, quiet=False):
    "Helper method for handling agent crashes"
    if not quiet: traceback.print_exc()
    self.gameOver = True
    self.agentCrashed = True
    self.rules.agentCrash(self, agentIndex)

  OLD_STDOUT = None
  OLD_STDERR = None

  def mute(self):
    if not self.muteAgents: return
    global OLD_STDOUT, OLD_STDERR
    import cStringIO
    OLD_STDOUT = sys.stdout
    OLD_STDERR = sys.stderr
    sys.stdout = cStringIO.StringIO()
    sys.stderr = cStringIO.StringIO()

  def unmute(self):
    if not self.muteAgents: return
    global OLD_STDOUT, OLD_STDERR
    sys.stdout.close()
    sys.stderr.close()
    # Revert stdout/stderr to originals
    sys.stdout = OLD_STDOUT
    sys.stderr = OLD_STDERR

  def getCostOfActions(self, actions):
    """
    Returns the cost of a particular sequence of actions.  If those actions
    include an illegal move, return 999999
    """
    agentIndex = self.startingIndex
    if actions == None: return 999999
    x,y= self.state.data.agentStates[ agentIndex ].getPosition()
    
    cost = 0
    for action in actions:
      # Check figure out the next state and see whether its' legal
      dx, dy = Actions.directionToVector(action)
      #print "dx , dy: ", dx , " , " , dy
      x, y = int(x + dx), int(y + dy)
      #print "x,y: ", x , " , ", y
      #if self.state.data.layout.walls[x][y]: return 999999
      cost += self.costFn((x,y))
      #print "cost: ", cost
    return cost




############################################################################################################################################################################
#                                   ************** HERE IS WHERE MY CODE STARTS **************
############################################################################################################################################################################


  # precondition: 
  # give a number 0-4 

  # postcondition: 
  # converts 0-4 into an agent's action which is its movement to direction -> north, south, west, east, or stop

  # 0 = stop
  # 1 = north
  # 2 = south
  # 3 = west
  # 4 = east

  def numToDirection(self, num):
    if num == 1:
      return Directions.NORTH
    if num == 2:
      #action = Directions.SOUTH
      return Directions.SOUTH
    if num == 3:
      #action = Directions.WEST
      return Directions.WEST
    if num == 4:
      #action = Directions.EAST
      return Directions.EAST
    if num == 0:
      #action = Directions.STOP
      return Directions.STOP

  
  # precondition: 

  # agentStates = current agent's state which is each agent's location 
  # agentGoals = the agent's goal locations
  #   -- agentStates is a Dictionary with the keys: "pacman", "ghost1", "ghost2"
  #   -- the values to each key: "pacman" = pacman's location (x,y), "ghost1" = ghost1's location (x,y), "ghost2" = ghost2's location (x,y)

  # postcondition:

  # successors = all legal successors for the current agentState
  # ** This is determined by evaluating every neighboring location to each agent  
  # ** Each agent has a possible 5 moves: STOP, NORTH, SOUTH, WEST, and EAST
  # ** If the next location is not:
  # **               the same location as a wall (obstacle) or the same location of another agent's location ( to prvevent collisions)
  # **               or the same location as another agent's next location ( so they do not move to the same location at the same time step )
  # then that successor that contains the next location for each agent : pacman, ghost1, and ghost2 is added to a list for each agent
  # the list for each agent: pacman
  #   pacmanSuccessorState = pacman's next location (x,y)
  #   pacmanAction = pacman's direction to the next location [direction] -> NORTH, WEST, EAST, SOUTH, STOP
  #   pacmanCost = 1.0 
  # the list for each agent: ghost1
  #   ghost1SuccessorState = ghost1's next location (x,y)
  #   ghost1Action = ghost1's direction to the next location [direction] -> NORTH, WEST, EAST, SOUTH, STOP
  #   ghost1Cost = 1.0
  # the list for each agent: ghost2
  #   ghost2SuccessorState = ghost2's next location (x,y)
  #   ghost2Action = ghost2's direction to the next location [direction] -> NORTH, WEST, EAST, SOUTH, STOP
  #   ghost2Cost = 1.0
 

  def getSuccessors(self, agentStates, agentGoals):
    # each agent's location -> (x,y)  
    pacmanLocation = agentStates["pacman"]
    ghost1Location = agentStates["ghost1"]
    ghost2Location = agentStates["ghost2"]

    # (x,y) for each agent's location
    x1,y1 = pacmanLocation
    x2,y2 = ghost1Location
    x3,y3 = ghost2Location
   
    # initializing a list for all of the successors   
    successors = []
    # creating 125 possible moves before judging legality: 5^3 moves 
    # (0,0,0,0), (0,0,01), ... (4,4,4,4)
    # each number represents a direction from the function I wrote: numToDirection
    successor = list(itertools.product([0,1,2,3,4], repeat = 3))
    
    # i = each possible move in the list created with the length 125
    for i in successor:
      pacman = i[0]
      ghost1 = i[1]
      ghost2 = i[2]

      # converting the # ( 0 - 4 ) into an action for each agent
      pacmanAction = self.numToDirection(pacman)
      ghost1Action = self.numToDirection(ghost1) 
      ghost2Action = self.numToDirection(ghost2)

      # appending the list of moves into a single list
      #listOfActions = [pacmanAction, ghost1Action, ghost2Action]
  
      # computing the new location for each agent's next location ( the child node )
      dx1, dy1 = Actions.directionToVector(pacmanAction)
      dx2, dy2 = Actions.directionToVector(ghost1Action)
      dx3, dy3 = Actions.directionToVector(ghost2Action)

      (nextX1 , nextY1) = int(x1 + dx1), int(y1 + dy1)
      (nextX2, nextY2) = int(x2 + dx2), int(y2 + dy2)
      (nextX3, nextY3) = int(x3 + dx3), int(y3 + dy3)

      # the child nodes:
      # pacman = pacmanSuccessorState
      # ghost1 = ghost1SuccessorState 
      # ghost2 = ghost2SuccessorState  
      pacmanSuccessorState = (nextX1 , nextY1)
      ghost1SuccessorState = (nextX2, nextY2)
      ghost2SuccessorState = (nextX3, nextY3)


      pacmanCost = self.costFn(pacmanSuccessorState)
      ghost1Cost = self.costFn(ghost1SuccessorState)
      ghost2Cost = self.costFn(ghost2SuccessorState)

      # nextState isa Dictionary containing the keys:
      # "pacman", "ghost1", "ghost2"
      # the values for each key are stored into a Dictionary as well

      # for each agent the values are stored as:
      # "childNode" -> agent's next location (x,y)
      # "direction" -> action's action to the child node: NORTH, SOUTH, WEST, EAST, STOP
      # "cost" -> 1.0
      nextState = {
      "pacman": {"childNode": pacmanSuccessorState, "direction": pacmanAction, "cost": pacmanCost},
      "ghost1": {"childNode": ghost1SuccessorState, "direction": ghost1Action, "cost": ghost1Cost},
      "ghost2": {"childNode": ghost2SuccessorState, "direction": ghost2Action, "cost": ghost2Cost}
      }

      successors.append(nextState)

    return successors

  
  # precondition: 

  # agentLocations = agent's current state 
  # a Dictionary with the keys: 'pacman', 'ghost1', 'ghost2'
  # each value for the key is a (x,y) current location

  # goalPositions = each agent's goals
  # a Dictionary with the keys: 'pacman', 'ghost1', 'ghost2' 
  # each value for the key is a (x,y) goal location

  def heuristicFunction( self, agentLocations, goalPositions):

    pacmanLocation = agentLocations['pacman']
    ghost1Location = agentLocations['ghost1']
    ghost2Location = agentLocations['ghost2']
    agent1 = manhattanDistance(pacmanLocation, goalPositions['pacman'])
    agent2 = manhattanDistance(ghost1Location, goalPositions['ghost1'])
    agent3 = manhattanDistance(ghost2Location, goalPositions['ghost2'])

    return max(agent1, agent2, agent3)
   

  # precondition: successors, closedList, currentStae, agentGoals

  # successors = a list of Dictionaries that contain every possible move for each agent: 125 moves
  # closedList = a list of previous locations that calculate how to get to the current location 
  # currentState = a Dictionary with the keys: 'pacman', 'ghost1', 'ghost2'

  # postcondition: prunedSuccessors 

  # prunedSuccessors = a Dictionary with the keys:
  # pacman, ghost1, ghost2
  # pacman = pacmanSuccessor -> pacman's next location (x,y)
  # ghost1 = ghost1Successor -> ghost1's next location (x,y)
  # ghost2 = ghost2Successor -> ghost2's next location (x,y)

  def shouldPruneSuccessor(self, successors, closedList, currentState, agentGoals):
    

    prunedSuccessors = []
    walls = self.state.data.layout.walls

    # a list of vectors that each agent has traveled to get to the current state
    pacmanPath = closedList['pacman']
    ghost1Path = closedList['ghost1']
    ghost2Path = closedList['ghost2']

    for successor in successors:
      pacmanSuccessor = successor['pacman']
      ghost1Successor = successor['ghost1']
      ghost2Successor = successor['ghost2']

      pacmanLocation = currentState['pacman']
      ghost1Location = currentState['ghost1']
      ghost2Location = currentState['ghost2']

      pacmanChildNode = pacmanSuccessor['childNode']
      ghost1ChildNode = ghost1Successor['childNode']
      ghost2ChildNode = ghost2Successor['childNode']


      pacmanDirection = pacmanSuccessor['direction']
      ghost1Direction = ghost1Successor['direction']
      ghost2Direction = ghost2Successor['direction']

      (pacmanX, pacmanY) = pacmanChildNode
      (ghost1X, ghost1Y) = ghost1ChildNode
      (ghost2X, ghost2Y) = ghost2ChildNode
        
      # if the next move for each agent does not equal to a wall (obstacle)
      if not ((walls[pacmanX][pacmanY] or walls[ghost1X][ghost1Y] or walls[ghost2X][ghost2Y])):
        
       
        # if the next move for each agent does not equal the other agent's childNodes or current location (preventing collisions)
        if not ((pacmanChildNode == ghost1ChildNode) or (ghost1ChildNode == ghost2ChildNode) or (pacmanChildNode == ghost2ChildNode )) :
          if not (ghost1ChildNode == pacmanLocation):
            if not (ghost1ChildNode == ghost2Location):
              if not (ghost2ChildNode == pacmanLocation):
                if not (pacmanChildNode == ghost2Location):
                          #if not ((ghost1ChildNode == pacmanLocation) and (ghost1ChildNode == ghost2Location)):
                            #if not ((ghost2ChildNode == pacmanLocation) and (ghost2ChildNode == ghost1Location)):
                              #if not ((pacmanChildNode == ghost1Location) and (pacmanChildNode == ghost2Location)):
                  

                  # if any of the agent's next move equals their respective goal location then create a child node
                  if (pacmanChildNode == agentGoals['pacman']) or (ghost1ChildNode == agentGoals['ghost1']) or (ghost2ChildNode == agentGoals['ghost2']):

                              childNode = {
                                          "pacman": pacmanSuccessor,
                                          "ghost1": ghost1Successor,
                                          "ghost2": ghost2Successor
                                          }
                              prunedSuccessors.append(childNode)

                  # if none of the child nodes have been recently visited already 
                  if not pacmanChildNode in pacmanPath:
                      if not ghost1ChildNode in ghost1Path:
                        if not ghost2ChildNode in ghost2Path:

                          # preventing the agents from passing the action 'Stop' for all agents
                          if not ((pacmanDirection == 'Stop') and (ghost1Direction == 'Stop') and (ghost2Direction == 'Stop')):
                            # as long as the agent is not at it's goal then keep searching.. if it is at its goal then stop
                            if not ( (pacmanLocation == agentGoals['pacman']) and (pacmanDirection == 'Stop') ):
                              if not ( (ghost1Location == agentGoals['ghost1']) and (ghost1Direction == 'Stop') ):
                                if not ( (ghost2Location == agentGoals['ghost2']) and (ghost2Direction == 'Stop') ):
                                  childNode = {
                                              "pacman": pacmanSuccessor,
                                              "ghost1": ghost1Successor,
                                              "ghost2": ghost2Successor
                                              }
                                  prunedSuccessors.append(childNode)

    return prunedSuccessors

    

  # precondition: agentPaths, agentGoals

  # agentPaths = a Dictionary that contains a list of every agent's path
  # the keys for agentPath: 'pacman', 'ghost1', 'ghost2'
  # the values to the keys = a List that is the path for each respective agent

  # agentGoals = a Dictionary that contains a location of every agent's goal location (x,y)
  # the keys for agentGoals: 'pacman', 'ghost1', 'ghost2'
  # the values to the keys = a location (x,y) on the grid

  # postcondition: returns a closedList
  # closedList = a Dictionary that has the keys: 'pacman', 'ghost1', 'ghost2'
  # the values to the keys = a list of (x,y) locations that contains the path for each agent

  def createClosedList(self, agentPaths, agentGoals):
    pacmanClosedList = []
    ghost1ClosedList = []
    ghost2ClosedList = []

    pacmanPath = agentPaths['pacman']
    ghost1Path = agentPaths['ghost1']
    ghost2Path = agentPaths['ghost2']

    pacmanGoal = agentGoals['pacman']
    ghost1Goal = agentGoals['ghost1']
    ghost2Goal = agentGoals['ghost2']

    pacmanStartingLocation = self.state.data.agentStates[0].getPosition()
    ghost1StartingLocation = self.state.data.agentStates[1].getPosition()
    ghost2StartingLocation = self.state.data.agentStates[2].getPosition()


    pacmanClosedList.append(pacmanStartingLocation)
    ghost1ClosedList.append(ghost1StartingLocation)
    ghost2ClosedList.append(ghost2StartingLocation)
    
    for i in pacmanPath:
      pacmanLocation = pacmanClosedList[-1]
      (x, y) = Actions.directionToVector(i)
      dx, dy = pacmanLocation

      pacmanNextLocation = (dx + x , dy + y )
      
      if pacmanLocation == pacmanGoal:
        pacmanNextLocation = pacmanLocation
      pacmanClosedList.append(pacmanNextLocation)

    for i in ghost1Path:
      
      ghost1Location = ghost1ClosedList[-1]
      (x, y) = Actions.directionToVector(i)
      dx, dy = ghost1Location

      ghost1NextLocation = (dx + x , dy + y )
      if ghost1Location == ghost1Goal:
        ghost1NextLocation = ghost1Location
      ghost1ClosedList.append(ghost1NextLocation)

    for i in ghost2Path:
     
      ghost2Location = ghost2ClosedList[-1]
      (x, y) = Actions.directionToVector(i)
      dx, dy = ghost2Location

      ghost2NextLocation = (dx + x , dy + y )
      if ghost2Location == ghost2Goal:
        ghost2NextLocation = ghost2Location
      ghost2ClosedList.append(ghost2NextLocation)

    closedList = {
    'pacman' : pacmanClosedList,
    'ghost1' : ghost1ClosedList,
    'ghost2' : ghost2ClosedList
    }

    return closedList


  # precondition: closedList 
  # closedList = a Dictionary that has the keys: 'pacman', 'ghost1', 'ghost2'
  # the values to the keys = a list of (x,y) locations that contains the path for each agent

  #postcondition: returns agentPaths
  # agentPaths is a Dictionary that has the keys: 'pacman', 'ghost1', 'ghost2'
  # the values to the keys = a list of [Directions] that create the agent paths
  # the agent's read their directions in the form of directions -> NORTH, SOUTH, WEST, EAST, STOP
  # so each key in the dictionary contains the path for that agent

  def buildPathFromClosedList(self, closedList):

    # a list for each agent that contains the previously visited locations from start to current location
    pacmanClosedList = closedList['pacman']
    ghost1ClosedList = closedList['ghost1']
    ghost2ClosedList = closedList['ghost2']

    # initalizing an empty list for each agent's path
    pacmanPath = []
    ghost1Path = []
    ghost2Path = []

    pacmanCycle = cycle(pacmanClosedList)
    ghost1Cycle = cycle(ghost1ClosedList)
    ghost2Cycle = cycle(ghost2ClosedList)

    pacmanNextLocation = pacmanCycle.next()
    ghost1NextLocation = ghost1Cycle.next()
    ghost2NextLocation = ghost2Cycle.next()

    
    for pathIndex in range(len(pacmanClosedList)):
      pacmanLocation, pacmanNextLocation = pacmanNextLocation, pacmanCycle.next()
      # if the pathIndex equals the last location in the list then set the next location to the current location
      if (pathIndex == len(pacmanClosedList) - 1):
        pacmanNextLocation = pacmanLocation

      x, y = pacmanLocation
      dx, dy = pacmanNextLocation

      # converting the (x,y) locations into directions for pacman's path
      if dx - x > 0:
        pacmanPath.append(Directions.EAST)
      if dx - x < 0:
        pacmanPath.append(Directions.WEST)
      if dy - y > 0:
        pacmanPath.append(Directions.NORTH)
      if dy - y < 0:
        pacmanPath.append(Directions.SOUTH)
      if (dx == x) and (dy == y):
        pacmanPath.append(Directions.STOP)
        
    for pathIndex in range(len(ghost1ClosedList)):
      ghost1Location, ghost1NextLocation = ghost1NextLocation, ghost1Cycle.next()
      if pathIndex == len(ghost1ClosedList) - 1:
        ghost1NextLocation = ghost1Location
     
      x, y = ghost1Location
      dx, dy = ghost1NextLocation

      # converting the (x,y) locations into directions for ghost1's path
      if dx - x > 0:
        ghost1Path.append(Directions.EAST)
      if dx - x < 0:
        ghost1Path.append(Directions.WEST)
      if dy - y > 0:
        ghost1Path.append(Directions.NORTH)
      if dy - y < 0:
        ghost1Path.append(Directions.SOUTH)
      if (dx == x) and (dy == y):
        #print "ghost 1 is DONE"
        ghost1Path.append(Directions.STOP)

    for pathIndex in range(len(ghost2ClosedList)):
      ghost2Location, ghost2NextLocation = ghost2NextLocation, ghost2Cycle.next()
      if pathIndex == len(ghost2ClosedList) - 1:
        ghost2NextLocation = ghost2Location

      x, y = ghost2Location
      dx, dy = ghost2NextLocation

      # converting the (x,y) locations into directions for ghost2's path
      if dx - x > 0:
        ghost2Path.append(Directions.EAST)
      if dx - x < 0:
        ghost2Path.append(Directions.WEST)
      if dy - y > 0:
        ghost2Path.append(Directions.NORTH)
      if dy - y < 0:
        ghost2Path.append(Directions.SOUTH)
      if (dx == x) and (dy == y):
        #print "ghost 2 is DONE"
        ghost2Path.append(Directions.STOP)

    if ( len(pacmanPath) > (len(ghost1Path) and len(ghost2Path) ) ):
      if ( len(ghost1Path) <= len(ghost2Path) ):
       differenceInLength = len(pacmanPath) - len(ghost1Path)
       for i in range(differenceInLength):
        ghost1Path.append(Directions.STOP)
      if ( len(ghost2Path) <= len(ghost1Path) ):
        differenceInLength = len(pacmanPath) - len(ghost2Path)
        for i in range(differenceInLength):
         ghost2Path.append(Directions.STOP)

    if ( len(ghost1Path) > ( len(pacmanPath) and len(ghost2Path) ) ):
      if ( len(pacmanPath) <= len(ghost2Path) ):
        differenceInLength = len(ghost1Path) - len(pacmanPath)
        for i in range(differenceInLength):
          pacmanPath.append(Directions.STOP)
      if ( len(ghost2Path) <= len(pacmanPath) ):
        differenceInLength = len(ghost1Path) - len(ghost2Path)
        for i in range(differenceInLength):
          ghost2Path.append(Directions.STOP)

    if ( len(ghost2Path) > (len(ghost1Path) and len(pacmanPath) ) ):
      if ( len(pacmanPath) <= len(ghost1Path) ):
        differenceInLength = len(ghost2Path) - len(pacmanPath)
        for i in range(differenceInLength):
          pacmanPath.append(Directions.STOP)
      if ( len(ghost1Path) <= len(pacmanPath) ):
        differenceInLength = len(ghost2Path) - len(ghost1Path)
        for i in range(differenceInLength):
          ghost1Path.append(Directions.STOP)

    agentPaths = {
    'pacman': pacmanPath,
    'ghost1': ghost1Path,
    'ghost2': ghost2Path
    }

    return agentPaths

    
     
  # precondition: agentLocations and agentGoals

  # agentLocations = a Dictionary that has the keys: 'pacman', 'ghost1', 'ghost2'
  # each key contains a (x,y) location that is that agent's current location
  # this function is called at the agent's initial starting location

  # agentGoals = a Dictionary that has the keys: 'pacman', 'ghost1', 'ghost2'
  # each key contains a (x,y) location that is that agent's goal location

  # postcondition: returns testPath
  # testPath is where the createClosedList then the buildPathFromClosedList functions are called to compute the path for each agent
  # testPath = a Dictionary that contains the keys: 'pacman', 'ghost1', 'ghost2'
  # each key contains a List that is that agent's path from the starting location to the goal location
  # this is the final function for the path finding that gets called in the MAPFinderq function

  def aStarPathFinding(self, agentLocations, agentGoals):
     

    pacmanClosedList = []
    ghost1ClosedList = []
    ghost2ClosedList = []  
    closedList = {
      'pacman': pacmanClosedList,
      'ghost1': ghost1ClosedList,
      'ghost2': ghost2ClosedList
      }  


    pacmanPath = []
    ghost1Path = []
    ghost2Path = []

    agentPaths = {
    'pacman': pacmanPath,
    'ghost1': ghost1Path,
    'ghost2': ghost2Path
    } 

    start = {
    'pacman': agentLocations[0],
    'ghost1': agentLocations[1],
    'ghost2': agentLocations[2]
    }

    goal = {
    'pacman': agentGoals[0],
    'ghost1': agentGoals[1],
    'ghost2': agentGoals[2]
    }

    pacman = self.state.data.agentStates[0].getPosition()
    ghost1 = self.state.data.agentStates[1].getPosition()
    ghost2 = self.state.data.agentStates[2].getPosition()

    walls = self.state.data.layout.walls

    g_value = self.getCostOfActions(agentPaths['pacman'])
    h_value = self.heuristicFunction(start, goal)
    f_value = g_value + h_value

    fringe = PriorityQueue() 
    fringe.push((start, agentPaths), f_value)
    i = 0
    
    while not fringe.isEmpty():
      print " \n \n \n"
      i += 1  
      print "i: ", i

      currentNode, nodeActions = fringe.pop()

      print "current node: ", currentNode
      print "agent goals: ", goal
      #print "node actions: ", nodeActions

      g_value = self.getCostOfActions(nodeActions['pacman']) 
      h_value = self.heuristicFunction(currentNode, goal)
      f_value = g_value + h_value
      print "G_value:", g_value
      print "h_value: ", h_value
      print "F_value: ", f_value
      
      if currentNode == goal:
        print "A* ended with currentNode == goal:"
        newClosedList = self.createClosedList(nodeActions, goal)
        testPath = self.buildPathFromClosedList(newClosedList)
        return testPath

      # calling my custom functions here
      successors = self.getSuccessors(currentNode, agentGoals)
      newClosedList = self.createClosedList(nodeActions, goal)
      pruneSuccessors = self.shouldPruneSuccessor(successors, newClosedList, currentNode, goal)

      
      

      for successor in pruneSuccessors:
        pacmanSuccessor = successor['pacman']
        ghost1Successor = successor['ghost1']
        ghost2Successor = successor['ghost2']

        pacmanChildNode = pacmanSuccessor['childNode']
        ghost1ChildNode = ghost1Successor['childNode']
        ghost2ChildNode = ghost2Successor['childNode']

        pacmanDirection = pacmanSuccessor['direction']
        ghost1Direction = ghost1Successor['direction']
        ghost2Direction = ghost2Successor['direction']

        childNode = {
        'pacman': pacmanChildNode,
        'ghost1': ghost1ChildNode,
        'ghost2': ghost2ChildNode
        }

        ################ creating the agentPaths here ###################
        tempPacmanPath = []
        tempGhost1Path = []
        tempGhost2Path = []
        pacmanPath = nodeActions['pacman']
        tempPacmanPath = pacmanPath + [pacmanSuccessor['direction']]

        ghost1Path = nodeActions['ghost1']
        tempGhost1Path = ghost1Path + [ghost1Successor['direction']]

        ghost2Path = nodeActions['ghost2']
        tempGhost2Path = ghost2Path + [ghost2Successor['direction']]

        agentPaths = {
        'pacman': tempPacmanPath,
        'ghost1': tempGhost1Path,
        'ghost2': tempGhost2Path
        }
        #print "agentPaths: ", agentPaths

        ############## checking value of h' , g' , and f' ####################
        # h' = tempHeuristic
        tempHeuristic = self.heuristicFunction(childNode, goal)
        # g' = tempCost
        tempCost = self.getCostOfActions(agentPaths['pacman'])
        # f' = tempGoal
        tempGoal = tempCost + tempHeuristic

        #print "tempCost: ", tempCost
        #print "tempHeuristic: ", tempHeuristic
        #print "tempGoal: ", tempGoal


        fringe.push( (childNode, agentPaths), tempGoal)

    return []

  #####################################################################################    
  #                              CBS BEGINS here                                      #
  ##################################################################################### 

  
  def convertPathWithDirectionsToCords(self, agent, path, goal):
    agentNewPath = []
    start = self.state.data.agentStates[agent].getPosition()
    agentNewPath.append(start)


    for i in path:
      location = agentNewPath[-1]
      (x, y) = Actions.directionToVector(i)
      #print"(x,y): ", (x,y)
      dx, dy = location
      #print "(dx, dy): ", (dx, dy)
      #print"location: ", location

      nextLocation = (dx + x, dy + y)
      #print"next location: ", nextLocation

      #if nextLocation == goal:
      if location == goal:
        nextLocation = location
      agentNewPath.append(nextLocation)
      #print "agentNewPath: ", agentNewPath
    return agentNewPath


  def createPathWithCords(self, agentPaths, agentGoals):
    print "agentPaths in CORDS FUNCTION: ", agentPaths
    print "LENGTH PACMAN: ", len(agentPaths[0])
    print "LENGTH GHOST 1: ", len(agentPaths[1])
    print "LENGTH GHOST 2: ", len(agentPaths[2])

    pacmanNewPath = []
    ghost1NewPath = []
    ghost2NewPath = []

    pacmanPath = agentPaths[0]
    ghost1Path = agentPaths[1]
    ghost2Path = agentPaths[2]
    print"ghost2Path in createPathWithCords: ", ghost2Path

    pacmanGoal = agentGoals[0]
    ghost1Goal = agentGoals[1]
    ghost2Goal = agentGoals[2]

    pacmanStartingLocation = self.state.data.agentStates[0].getPosition()
    ghost1StartingLocation = self.state.data.agentStates[1].getPosition()
    ghost2StartingLocation = self.state.data.agentStates[2].getPosition()


    pacmanNewPath.append(pacmanStartingLocation)
    ghost1NewPath.append(ghost1StartingLocation)
    ghost2NewPath.append(ghost2StartingLocation)
    
    for i in pacmanPath:
      pacmanLocation = pacmanNewPath[-1]
      (x, y) = Actions.directionToVector(i)
      dx, dy = pacmanLocation

      pacmanNextLocation = (dx + x , dy + y )
      
      if pacmanLocation == pacmanGoal:
        pacmanNextLocation = pacmanLocation
      pacmanNewPath.append(pacmanNextLocation)

    for i in ghost1Path:
      
      ghost1Location = ghost1NewPath[-1]
      (x, y) = Actions.directionToVector(i)
      dx, dy = ghost1Location

      ghost1NextLocation = (dx + x , dy + y )
      if ghost1Location == ghost1Goal:
        ghost1NextLocation = ghost1Location
      ghost1NewPath.append(ghost1NextLocation)

    for i in ghost2Path:
     
      ghost2Location = ghost2NewPath[-1]
      (x, y) = Actions.directionToVector(i)
      dx, dy = ghost2Location

      ghost2NextLocation = (dx + x , dy + y )
      if ghost2Location == ghost2Goal:
        ghost2NextLocation = ghost2Location
      ghost2NewPath.append(ghost2NextLocation)

    newPath = {
    'pacman' : pacmanNewPath,
    'ghost1' : ghost1NewPath,
    'ghost2' : ghost2NewPath
    }

    return newPath

  def getSuccessor(self, agentLocation):
    #print "Entered getSuccessor function"
    successors = []


    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]:
      x,y = agentLocation
      dx, dy = Actions.directionToVector(action)
      nextx, nexty = int(x + dx), int(y + dy)
      if not self.state.data.layout.walls[nextx][nexty]:
          nextState = (nextx, nexty)
          cost = self.costFn(nextState)
          successors.append( ( nextState, action, cost) )

        
    # Bookkeeping for display purposes
    self._expanded += 1 
    if agentLocation not in self._visited:
      self._visited[agentLocation] = True
      self._visitedlist.append(agentLocation)
    #print "successors: ", successors  
    return successors

  def getSuccessorWithConstraints(self, agentLocation, constraints, t):
    print "Entered getSuccessorWithConstraints function"
    successors = []
    item = []
    #print "constraints in getSuccessorWithConstraints: ", constraints
    #print "timeStep: ", t
    if item in constraints:
      #print"constraints has an empty '[]' in it!"
      constraints.remove([])
    #print "constraints removed the []?: ", constraints
    constraintTable = []

    for each_constraint in constraints:
      tempList = []
      tempList.append(each_constraint[1])
      tempList.append(each_constraint[2])
      constraintTable.append(tempList)
    print"constraintTable in getSuccessorWithConstraints: ", constraintTable
    


    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]:
      x,y = agentLocation
      dx, dy = Actions.directionToVector(action)
      nextx, nexty = int(x + dx), int(y + dy)
      agentNextLocation = (nextx, nexty)

      if not self.state.data.layout.walls[nextx][nexty]:
        pair = [agentNextLocation, t+1]
        print "pair: ", pair
        if pair in constraintTable:
          print"found a pair that is a conflict successor!"
        if not pair in constraintTable:
          #if not ( ( (nextx, nexty) == i[1]) and (t == i[2]) ):
          nextState = (nextx, nexty)
          cost = self.costFn(nextState)
          successors.append( ( nextState, action, cost) )

        
    # Bookkeeping for display purposes
    self._expanded += 1 
    if agentLocation not in self._visited:
      self._visited[agentLocation] = True
      self._visitedlist.append(agentLocation)
    #print "successors: ", successors  
    return successors

  def singleAgentAstarSearch( self, agentStart, agentGoal):
    #print "ENTERED singleAgentAstarSearch ******"
    fringe = PriorityQueue() 
    closedList = []
    start = agentStart
    goal = agentGoal
    h_value = manhattanDistance(start, goal)
    timeStep = 0
    fringe.push( (start, []), h_value )
    while not fringe.isEmpty():
      #print "agent starting location: ", agentStart
      currentNode, nodeActions = fringe.pop()
      #print "currentNode in search: ", currentNode
      if currentNode == goal:
        return nodeActions
      closedList.append(currentNode)
      successors = self.getSuccessor(currentNode)
      #print "successors: ", successors
      for childNode, direction, cost in successors:
        if not childNode in closedList:
          tempCost = nodeActions + [direction]
          tempGoal = self.getCostOfActions(tempCost) + manhattanDistance(childNode, goal)
          fringe.push((childNode, tempCost), tempGoal)

    return []

  def singleAgentAstarSearchWithConstraints( self, agent, agentStart, agentGoal, constraints):
    #print "ENTERED singleAgentAstarSearchWithConstraints ******"
    print "agent: ", agent 
    print "constraints in singleAgentAstarSearchWithConstraints: ", constraints
    #print "constraints[0]: ", constraints[0]
    
    fringe = PriorityQueue() 
    closedList = []
    start = agentStart
    goal = agentGoal
    h_value = manhattanDistance(start, goal)
    timeStep = 0
    fringe.push( (start, []), h_value )

    while not fringe.isEmpty():
      #print "agent starting location: ", agentStart
      currentNode, nodeActions = fringe.pop()
      #print "currentNode in search: ", currentNode
      
      if currentNode == goal:
        print"currentNode == goal"
        print "nodeActions after testing? ", nodeActions
        return nodeActions
        
      closedList.append(currentNode)
      #print "time step: ", len(nodeActions)
      timeStep = len(nodeActions)
      nextTimeStep = timeStep + 1
      successors = self.getSuccessorWithConstraints(currentNode, constraints, timeStep)
      print "\n \n"
      #print "successors: ", successors
      for childNode, direction, cost in successors:
        #print "*len(constraints) inside successor for-loop: ", len(constraints)
        if not childNode in closedList:
          for agent_ai, conflict_location, t in constraints:
            #print "constraints: ", constraints
            #print "len(constraints): ", len(constraints)
            #print "agent_ai: ", agent_ai, "conflict_location: ", conflict_location, "t: ", t
            if not ((childNode == conflict_location) and (nextTimeStep == t)):
              tempCost = nodeActions + [direction]
              tempGoal = self.getCostOfActions(tempCost) + manhattanDistance(childNode, goal)
              fringe.push((childNode, tempCost), tempGoal)
              #print "childNode = ", childNode, "location of conflict = ", constraint[1]
              #print "timeStep in singleAgentAstarSearch: ", timeStep
              #print "nodeActions: ", nodeActions
              #print "timeStep in constraint: ", constraint[2]


    return []

  def hasNoConflicts(self, agentPaths, agentGoals):
    timeStep = 0

    # converting the paths from [directions] to (x,y) cords so it is easier to check for collisions
    cord_paths = self.createPathWithCords(agentPaths, agentGoals)
    # new paths with x-y cords
    pacmanPath = cord_paths['pacman']
    ghost1Path = cord_paths['ghost1']
    ghost2Path = cord_paths['ghost2']
    print "ghost2Path: ", ghost2Path
    print "len(ghost2Path): ", len(ghost2Path)

    timeStep = 0
    # initial starting position
    pacmanLocation = pacmanPath[timeStep]
    ghost1Location = ghost1Path[timeStep]
    ghost2Location = ghost2Path[timeStep]

    for i in range(len(ghost2Path)):
      # updating location each iteration in loop
      pacmanLocation = pacmanPath[timeStep]
      ghost1Location = ghost1Path[timeStep]
      ghost2Location = ghost2Path[timeStep]

      # output
      #print "time step: ", timeStep
      #print "pacman location: ", pacmanLocation
      #print "ghost1 location: ", ghost1Location
      #print "ghost2 location: ", ghost2Location
      #print "\n"

      # if pacman's location equals the location of ghost1 at timeStep(t)
      if (pacmanLocation == ghost1Location):
        return False
      # if pacman's location equals the location of ghost2 at timeStep(t)
      if (pacmanLocation == ghost2Location):
        return False
      # if ghost1's location equals the location of ghost2 at timeStep(t)
      if (ghost1Location == ghost2Location):
        return False
      # increment the timeStep each move
      timeStep += 1


    # return true is no conflicts occured between the agents
    return True

  def findConflictsInPath(self, agentPaths, agentGoals):
    timeStep = 0

    cord_paths = self.createPathWithCords(agentPaths, agentGoals)
    pacmanPath = cord_paths['pacman']
    ghost1Path = cord_paths['ghost1']
    ghost2Path = cord_paths['ghost2']

    pacmanLocation = self.state.data.agentStates[ 0 ].getPosition()
    ghost1Location = self.state.data.agentStates[ 1 ].getPosition()
    ghost2Location = self.state.data.agentStates[ 2 ].getPosition()

    conflicts = []

    pacman = 0
    ghost1 = 1
    ghost2 = 2

    #print "pacman's id: ", pacman
    #print "ghost1's id: ", ghost1
    #print "ghost2's id: ", ghost2

    
    for i in range(len(pacmanPath)):
      pacmanLocation = pacmanPath[timeStep]
      ghost1Location = ghost1Path[timeStep]
      ghost2Location = ghost2Path[timeStep]

      #pacmanNextLocation = pacmanPath[timeStep+1]
      #ghost1NextLocation = ghost1Path[timeStep+1]
      #ghost2NextLocation = ghost2Path[timeStep+1]

      # if pacman's location equals the location of ghost1
      if (pacmanLocation == ghost1Location):
        # C = (ai, aj, v, t)
        conflict = (pacman, ghost1, pacmanLocation, timeStep)
        print "conflict occured: ", conflict
        conflicts.append(conflict)

      if (pacmanLocation == ghost2Location):
        # C = (ai, aj, v, t)
        conflict = (pacman, ghost2, pacmanLocation, timeStep)
        print "conflict occured: ", conflict
        conflicts.append(conflict)

      if (ghost1Location == ghost2Location):
        # C = (ai, aj, v, t)
        conflict = (ghost1, ghost2, ghost1Location, timeStep)
        print "conflict occured: ", conflict
        conflicts.append(conflict)

      timeStep += 1
    return conflicts

  def makePathsTheSameLength(self, agentPaths):
    #print "makePathsTheSameLength ENTER"
    #print "agentPaths: ", agentPaths
    pacmanPath = agentPaths[0]
    ghost1Path = agentPaths[1]
    ghost2Path = agentPaths[2]

    pacmanLength = len(pacmanPath)
    ghost1Length = len(ghost1Path)
    ghost2Length = len(ghost2Path)

    maxLength = max(pacmanLength, ghost1Length, ghost2Length)
    #print "maxLength: ", maxLength

    #if pacmanLength < maxLength:
      #pacmanPath.append(Directions.STOP)

    for agentPath in agentPaths:
      #print "len(agentPath): ", len(agentPath)
      if len(agentPath) < maxLength:
        differenceInLength = maxLength - len(agentPath)
        for i in range(differenceInLength):
          agentPath.append(Directions.STOP)

    pacmanLength = len(pacmanPath)
    ghost1Length = len(ghost1Path)
    ghost2Length = len(ghost2Path)

    #print "actual pacman length: ", pacmanLength
    #print "actual ghost1 length: ", ghost1Length
    #print "actual ghost2 legnth: ", ghost2Length
    return agentPaths

  def singleAgentSearchAfterBackChain(self, agent, start, goal, constraints):
    #print "ENTERED singleAgentAstarSearchWithConstraints ******"
    print "agent: ", agent 
    print "constraints in singleAgentSearchAfterBackChain: ", constraints
    
    constraintTable = []

    for each_constraint in constraints:
      tempList = []
      tempList.append(each_constraint[1])
      tempList.append(each_constraint[2])
      constraintTable.append(tempList)
    print"constraintTable: ", constraintTable
    
    fringe = PriorityQueue() 
    closedList = []
    h_value = manhattanDistance(start, goal)
    timeStep = 0
    fringe.push( (start, []), h_value )

    z = 1
    while not fringe.isEmpty():
      print "z = ", z
      #print "agent starting location: ", agentStart
      currentNode, nodeActions = fringe.pop()
      #print "currentNode in search: ", currentNode

      
      if currentNode == goal:
        print"currentNode == goal"
        print "nodeActions after testing? ", nodeActions
        return nodeActions
        
      closedList.append(currentNode)
      #print "time step: ", len(nodeActions)
      timeStep = len(nodeActions)
      nextTimeStep = timeStep + 1
      successors = self.getSuccessorWithConstraints(currentNode, constraints, timeStep)
      print "successors: ", successors
      print "\n \n"
      #print "successors: ", successors
      for childNode, direction, cost in successors:
        #print "*len(constraints) inside successor for-loop: ", len(constraints)
        test = [childNode, timeStep+1]
        #print "currentNode: ", currentNode, "childNode: ", childNode, "timeStep: ", timeStep, "nextTimeStep: ", timeStep+1
        #print "test LOOK HERE: ", test
        if test in constraintTable:
          print "found a child node that is a constraint: ", test 
        if not (test in constraintTable):
          print"test is not in constraintTable: ", test 
          tempCost = nodeActions + [direction]
          tempGoal = self.getCostOfActions(tempCost) + manhattanDistance(childNode, goal)
          fringe.push((childNode, tempCost), tempGoal)
          #print"MAYBE????"
          #print"test that is maybe?: ", test
          #if not childNode in closedList:
            #for agent_ai, conflict_location, t in constraints:
            #for index in constraintTable:
              #if not ((childNode == index[0]) and (nextTimeStep == index[1])):
                #print "childNode: ", childNode, "constraintLocation: ", index[0],"timeStep: ", timeStep, "nextTimeStep: ", nextTimeStep, "t in constraint: ", index[1]
                



    return []


  def reconstructPathWithConstraints(self, agent, start, goal, constraints, solution):
    #print "*** ENTERED reconstructPathWithConstraints ****"
    #print "constraints: ", constraints
    #print "constraints[0]: ", constraints[0]
    #print "start: ", start
    #print "v: ", v
    #print "goal: ", goal
    #print "solution: ", solution

    #timeStep = constraints[2]
    agentSolution = solution[agent]

    print"constraints in reconstructPathWithConstraints: ", constraints

    agentSolution = self.singleAgentAstarSearchWithConstraints(agent, start, goal, constraints)

    #print "agent solution: ", agentSolution
    #print "len(agentSolution): ", len(agentSolution)
    solution[agent] = agentSolution
    #print "solution: ", solution

    solution = self.makePathsTheSameLength(solution)

    return solution

  
  def doesPathRequireBackChaining(self, agent, path, constraint):
    print "path: ", path
    pathLength = len(path)
    index = pathLength-1
    constraintList = []
    start = path[0]
    end = path[-1]
    print "constraints: ", constraint
    #print"path[1]: ", path[1]
    #print"constraint in checkSingleAgentPathForBackChaining: ", constraint
    if len(constraint) > 0:
      conflict_location = constraint[1]



    #print"start in checkSingleAgentPathForBackChaining: ", start  
    #print"conflict_location in checkSingleAgentPathForBackChaining: ", conflict_location

    # see where chain ends
    if start == conflict_location:
      return False
    for i in range(pathLength):
      #print"i in pathLength: ", i
      #print"index in pathLength: ", index
      agentLocation = path[index]

      if index == 1:
        
        #print "agentLocation", agentLocation
        nextAgentLocation = path[index-1]
        #print "nextAgentLocation: ", nextAgentLocation
      if index != 0:
        nextAgentLocation = path[index-1]


        if agentLocation == nextAgentLocation:
          return True 
    return False



  def checkSingleAgentPathForBackChaining(self, agent, path, constraint):
    print "path: ", path
    pathLength = len(path)
    index = pathLength-1
    constraintList = []
    start = path[0]
    end = path[-1]
    print "constraints: ", constraint
    #print"path[1]: ", path[1]
    #print"constraint in checkSingleAgentPathForBackChaining: ", constraint
    if len(constraint) > 0:
      conflict_location = constraint[1]



    #print"start in checkSingleAgentPathForBackChaining: ", start  
    #print"conflict_location in checkSingleAgentPathForBackChaining: ", conflict_location

    # see where chain ends
    if start != conflict_location:
      for i in range(pathLength):
        #print"i in pathLength: ", i
        #print"index in pathLength: ", index
        agentLocation = path[index]

        if index == 1:
          
          #print "agentLocation", agentLocation
          nextAgentLocation = path[index-1]
          #print "nextAgentLocation: ", nextAgentLocation
        if index != 0:
          nextAgentLocation = path[index-1]


          if agentLocation == nextAgentLocation:
            #print"path in checkSingleAgentPathForBackChaining: ", path
            #print"index in checkSingleAgentPathForBackChaining: ", index
            #print"agentLocation in checkSingleAgentPathForBackChaining: ", agentLocation
            #print"nextAgentLocation in checkSingleAgentPathForBackChaining: ", nextAgentLocation
            constraint = (agent, path[index], index-1)
            #print"Back chaining has spotted a new constraint: ", constraint
            constraintList.append(constraint)
          
      
        index-=1
    print "constraintList in checkSingleAgentPathForBackChaining: ", constraintList
    return constraintList

  def checkPathsForBackChaining(self, agentPaths, constraints):
    # dont need to loop thru each index of the paths anymore, just inspect the index of the constraint and back chain that agent's path
    #print"entered checkPathsForBackChaining **"
    pacmanPath = agentPaths[0]
    ghost1Path = agentPaths[1]
    ghost2Path = agentPaths[2]

    #print "constraints in checkPathsForBackChaining: ", constraints

    back_chain_constraints = constraints

    for constraint in constraints:
      agent = constraint[0]
      location = constraint[1]
      timeStep = constraint[2]

      #print "agent: ", agent, "location: ", location, "timeStep: ", timeStep
      agentPathToBackChain = agentPaths[agent]
      conflict_location = agentPathToBackChain[timeStep]
      #print "agentPathToBackChain: ", agentPathToBackChain
      #print "conflict_location: ", conflict_location



      # back chain the path now
      t = timeStep
      index = conflict_location
      for i in range(timeStep):
        #print"t = ", t
        index = agentPathToBackChain[t]
        #print "index: ", index

        if ((t == 0) and (index == agentPaths[0])):
          print "prune this node"

        if t != 0:
          next_index = agentPathToBackChain[t-1]
          #print "next_index: ", next_index
          if index == next_index:
            #print "adding new backchain constraint!"
            new_constraint = (agent, index,t)
            #print "new_constraint: ", new_constraint
            back_chain_constraints.append(new_constraint)
        t-=1
      #print "back_chain_constraints: ", back_chain_constraints
      return back_chain_constraints
    return constraints

  def converLocationToDirection(self, agentLocation, nextAgentLocation):
    x,y = agentLocation
    x1,y1 = nextAgentLocation

    diff_x = x1 - x
    diff_y = y1 - y

    #print "diff_x: ", diff_x
    #print "diff_y: ", diff_y

    if diff_x == -1:
      return Directions.WEST
    if diff_x == 1:
      return Directions.EAST
    if diff_y == 1:
      return Directions.NORTH
    if diff_y == -1:
      return Directions.SOUTH
    if agentLocation == nextAgentLocation:
      return Directions.STOP

  def convertAgentPathToDirections(self, path):
    #print "path in convertAgentPathToDirections: ", path
    pathLength = len(path) - 1
    #print "the path length is: ", pathLength
    new_path = []

    for i in range(pathLength):
      #print "the i in convertAgentPathToDirections = ", i
      agentLocation = path[i]
      #print "agentLocation in convertAgentPathToDirections: ", agentLocation
      if (i <= pathLength-1):
        agentNextLocation = path[i+1]
        #print "agentNextLocation in convertAgentPathToDirections: ", agentNextLocation
        agentDirection = self.converLocationToDirection(agentLocation, agentNextLocation)
        #print "agentDirection in convertAgentPathToDirections: ", agentDirection
        new_path.append(agentDirection)
    print "new_path in convertAgentPathToDirections: ", new_path
    return new_path

  def convertAgentPathsToDirections(self, agentPaths):
    print "entered convertAgentPathsToDirections **"
    print "agentPaths: ", agentPaths

    pacmanPath = agentPaths[0]
    ghost1Path = agentPaths[1]
    ghost2Path = agentPaths[2]

    new_pacmanPath = []
    new_ghost1Path = []
    new_ghost2Path = []

    

    pathLength = len(pacmanPath)

    #if pathLength == 0:
    for i in range(pathLength):
      print "i == ", i
      pacmanLocation = pacmanPath[i]
      ghost1Location = ghost1Path[i]
      ghost2Location = ghost2Path[i]
      


      if (i <= pathLength -1):
        pacmanNextLocation = pacmanPath[i+1]
        ghost1NextLocation = ghost1Path[i+1]
        ghost2NextLocation = ghost2Path[i+1]

      #print "pacmanLocation: ", pacmanLocation
      #print "ghost1Location: ", ghost1Location
      #print "ghost2Location: ", ghost2Location
      #print "\n"
      #print "pacmanNextLocation: ", pacmanNextLocation
      #print "ghost1NextLocation: ", ghost1NextLocation
      #print "ghost2NextLocation: ", ghost2NextLocation

      pacmanDirection = self.converLocationToDirection(pacmanLocation, pacmanNextLocation)
      ghost1Direction = self.converLocationToDirection(ghost1Location, ghost1NextLocation)
      ghost2Direction = self.converLocationToDirection(ghost2Location, ghost2NextLocation)
      #print "pacmanDirection: ", pacmanDirection
      #print "ghost1Direction: ", ghost1Direction
      #print "ghost2Direction: ", ghost2Direction

      new_pacmanPath.append(pacmanDirection)
      new_ghost1Path.append(ghost1Direction)
      new_ghost2Path.append(ghost2Direction)
    solution = [new_pacmanPath, new_ghost1Path, new_ghost2Path]

    return solution



  
  def singleAgentRecoveryMethod(self, agent, path, constraints, goal):
    print "entered singleAgentRecoveryMethod for agent ", agent
    print "agent: ", agent, "path: ", path, "constraints: ", constraints, "goal: ", goal 
    start = path[0]
    pathLength = len(path)
    counter = pathLength
    location = path[-1]
    constraint = constraints[0]

    back_chain_constraints = self.checkSingleAgentPathForBackChaining(agent, path, constraint)
    print "back_chain_constraints: ", back_chain_constraints

    new_path = self.singleAgentSearchAfterBackChain(agent, start, goal, back_chain_constraints)
    print "new_path nigga: ", new_path
    data = [new_path, back_chain_constraints]
    return data



  def enterRecoveryMethod(self, agent, agentPaths, constraints, goals):

    pacman = 0
    ghost1 = 1
    ghost2 = 2
    print "agent in enterRecoveryMethod: ", agent

    initialPositions = []
    for n in range(len(self.agents)):
      initialPositions.append(self.state.data.agentStates[ n ].getPosition())

    pacmanPath = agentPaths['pacman']
    ghost1Path = agentPaths['ghost1']
    ghost2Path = agentPaths['ghost2']

    print "constraints in enterRecoveryMethod: ", constraints
    print "ENTERED enterRecoveryMethod "
    #print "agentPaths: ", agentPaths
    print "pacmanPath in enterRecoveryMethod: ", pacmanPath
    print "ghost1Path in enterRecoveryMethod: ", ghost1Path
    print "ghost2Path in enterRecoveryMethod: ", ghost2Path

    pathLength = len(pacmanPath)
    counter = pathLength
    pacmanLocation = pacmanPath[-1]
    ghost1Location = ghost1Path[-1]
    ghost2Location = ghost2Path[-1]

    pacmanConstraints = []
    ghost1Constraints = []
    ghost2Constraints = []


    for index in range(pathLength):
      #print "counter: ", counter
      pacmanLocation = pacmanPath[counter-1]
      ghost1Location = ghost1Path[counter-1]
      ghost2Location = ghost2Path[counter-1]
      if pacmanLocation == ghost1Location:
        #print "pacman and ghost1 conflict!"
        #if not pacmanLocation == pacmanPath[0]:
        pacmanConstraint = (pacman, pacmanLocation, counter-1)
        if pacmanConstraint not in pacmanConstraints:
          pacmanConstraints.append(pacmanConstraint)
        pacmanConstraint_test = self.checkSingleAgentPathForBackChaining(pacman, pacmanPath, pacmanConstraint)
        for i in pacmanConstraint_test:
          if i not in pacmanConstraints:
            pacmanConstraints.append(i)
        #if not ghost1Location == ghost1Path[0]:
        ghost1Constraint = (ghost1, ghost1Location, counter-1)
        if ghost1Constraint not in ghost1Constraints:
         ghost1Constraints.append(ghost1Constraint)
        ghost1Constraint_test = self.checkSingleAgentPathForBackChaining(ghost1, ghost1Path, ghost1Constraint)
        for k in ghost1Constraint_test:
          if k not in ghost1Constraints:
            ghost1Constraints.append(k)
            print "ghost1Constraints: ", ghost1Constraints

      if pacmanLocation == ghost2Location:
        #print "pacman and ghost2 conflict!"
        #if not pacmanLocation == pacmanPath[0]:
        pacmanConstraint = (pacman, pacmanLocation, counter-1)
        if pacmanConstraint not in pacmanConstraints:
          pacmanConstraints.append(pacmanConstraint)
        pacmanConstraint_test = self.checkSingleAgentPathForBackChaining(pacman, pacmanPath, pacmanConstraint)
        for i in pacmanConstraint_test:
          if i not in pacmanConstraints:
            pacmanConstraints.append(i)
        #if not ghost2Location == ghost2Path[0]:
        ghost2Constraint = (ghost2, ghost2Location, counter-1)
        if ghost2Constraint not in ghost2Constraints:
          ghost2Constraints.append(ghost2Constraint)
        ghost2Constraint_test = self.checkSingleAgentPathForBackChaining(ghost2, ghost2Path, ghost2Constraint)
        for k in ghost2Constraint_test:
          if k not in ghost2Constraints:
            ghost2Constraints.append(k)

      if ghost1Location == ghost2Location:
        #print "ghost1 and ghost2 conflict!"
        #print "ghost1Path: ", ghost1Path
        #print "ghost2Path: ", ghost2Path
        #print "ghost1Location: ", ghost1Location
        #print "ghost2Location: ", ghost2Location
        #print "index: ", index
        #print "counter-1: ", counter-1
        ghost1Constraint = (ghost1, ghost1Location, counter-1)
        if ghost1Constraint not in ghost1Constraints:
          ghost1Constraints.append(ghost1Constraint)
        ghost1Constraint_test = self.checkSingleAgentPathForBackChaining(ghost1, ghost1Path, ghost1Constraint)
        for i in ghost1Constraint_test:
          if i not in ghost1Constraints:
            ghost1Constraints.append(i)
        #if not ghost2Location == ghost2Path[0]:
        ghost2Constraint = (ghost2, ghost2Location, counter-1)
        if ghost2Constraint not in ghost2Constraints:
          ghost2Constraints.append(ghost2Constraint)
        ghost2Constraint_test = self.checkSingleAgentPathForBackChaining(ghost2, ghost2Path, ghost2Constraint)
        for k in ghost2Constraint_test:
          if k not in ghost2Constraints:
            ghost2Constraints.append(k)
        # this shows the conflict @ location (1,2)
        # back chain both agents
      #print "pacmanConstraints: ", pacmanConstraints
      #print "ghost1Constraints: ", ghost1Constraints
      #print "ghost2Constraints: ", ghost2Constraints
      counter-=1
    test = []
    #test = [pacmanConstraints, ghost1Constraints, ghost2Constraints]
    print "test: ", test
    #if len(pacmanConstraints) > 0:
    new_pacmanPath = self.singleAgentSearchAfterBackChain(pacman, initialPositions[pacman], goals[pacman], pacmanConstraints)
    #print "new_pacmanPath: ", new_pacmanPath
    #if len(ghost1Constraints) > 0:
    print "ghost1Constraints before new_ghost1Path: ", ghost1Constraints
    new_ghost1Path = self.singleAgentSearchAfterBackChain(ghost1, initialPositions[ghost1], goals[ghost1], ghost1Constraints)
    #print "new_ghost1Path: ", new_ghost1Path
    #if len(ghost2Constraints) > 0:
    new_ghost2Path = self.singleAgentSearchAfterBackChain(ghost2, initialPositions[ghost2], goals[ghost2], ghost2Constraints)
    #print "new_ghost2Path: ", new_ghost2Path

    print "new_pacmanPath in enterRecoveryMethod: ", new_pacmanPath
    print "new_ghost1Path in enterRecoveryMethod: ", new_ghost1Path
    print "new_ghost2Path in enterRecoveryMethod: ", new_ghost2Path

    #test = [pacmanConstraints, ghost1Constraints, ghost2Constraints]
    if agent == 0:
      print"enter agent == 0"
      #print "ghost1Path: ", ghost1Path
      ghost1Path = self.convertAgentPathToDirections(ghost1Path)
      
      ghost2Path = self.convertAgentPathToDirections(ghost2Path)
      #print "ghost2Path: ", ghost2Path
      solution = [new_pacmanPath, ghost1Path, ghost2Path]
      test = pacmanConstraints
      #cost = len(new_pacmanPath)

    if agent == 1:
      print"enter agent == 1"
      pacmanPath = self.convertAgentPathToDirections(pacmanPath)
      ghost2Path = self.convertAgentPathToDirections(ghost2Path)
      solution = [pacmanPath, new_ghost1Path, ghost2Path]
      test = ghost1Constraints
      print "solution for agent 1 from enterRecoveryMethod: ", solution
      #cost = len(new_ghost1Path)

    if agent == 2:
      print"enter agent == 2"
      pacmanPath = self.convertAgentPathToDirections(pacmanPath)
      ghost1Path = self.convertAgentPathToDirections(ghost1Path)
      solution = [pacmanPath, ghost1Path, new_ghost2Path]
      test = ghost2Constraints
      #cost = len(new_pacmanPath)

    for constraint in constraints:
      test.append(constraint)
    print "solution: ", solution
    solution = self.makePathsTheSameLength(solution)
    cost = len(solution[0])
    #print"NEWEST solution SEE IF PATHS STAY THE SAME: ", solution
    data = {'constraints':test, 'solution': solution, 'cost': cost}
    return data
    



  

  def conflictBasedSearch(self, agentPaths, agentGoals):
    timeStep = 0
    constraints = []
    goals = agentGoals
    solution = agentPaths
    cost = len(solution[0])
    
    data = {'cost':cost, 'constraints':constraints, 'solution':solution}
    CT = ConstraintTree(data)
    print(CT.getRootVal())
    
    
    fringe = PriorityQueue()
    fringe.push(CT, cost)

    initialPositions = []
    for n in range(len(self.agents)):
      initialPositions.append(self.state.data.agentStates[ n ].getPosition())


    m = 1
    while not fringe.isEmpty():
      p = fringe.pop()
      print "p = ", p
      test_data = p.getRootVal()
      print "test_data: ", test_data

      pSolution = test_data['solution']
      pConstraints = test_data['constraints']
      test = self.makePathsTheSameLength(pSolution)
      print"test: ", test 
      pCost = test_data['cost']
      print "\n \n "
      print "m = ", m
      print "pSolution: ", pSolution
      print "pConstraints: ", pConstraints
      print "pCost: ", pCost
      
      #pSolution_withCords = self.createPathWithCords(pSolution, goals)
      #print "pSolution_withCords: ", pSolution_withCords
      #new_pData = self.enterRecoveryMethod(pSolution_withCords, pConstraints)
      #print "new_pData: ", new_pData

      #pSolution = new_pData['solution']
      #pConstraints = new_pData['constraints']
      #test = pConstraints
      #pCost = new_pData['cost']
      

      # validate the paths in P until a conflict occurs
      # if hasNoConflicts(p, goals) == true, then p has no conflicts
      # if P has no conflicts then return P.solution bc P is goal
      if self.hasNoConflicts(pSolution, goals) == True:
        print "CBS SUCCESS!!!"
        #print "solution: ", p.data['solution']
        return pSolution
      
      

      # C <- first Conflict (ai, aj, v, t)
      conflicts = self.findConflictsInPath(pSolution, goals)
      print "conflict from cbs: ", conflicts

      # for each agent ai in C do
      #for ai, aj, v, t in conflicts: 
      for conflict in conflicts:
        agent_ai = conflict[0]
        agent_aj = conflict[1]
        agents = []
        agents.append(agent_ai)
        agents.append(agent_aj)
        print "agents in conflict: ", agents
        print "len agents: ", len(agents)
        #print "test constraints before ai in agents for loop: ", test
        jj = 1 
        for ai in agents:
          print "agents: ", agents
          print "ai: ", ai
          v = conflict[2]
          t = conflict[3]
          
          print "pConstraints: ", pConstraints
          #copying the pConstraitns by value not by reference
          aConstraints = pConstraints[:]
          #aConstraints.append((ai, v, t))
          testing = (ai,v,t)
          
          aConstraints.insert(0, testing)
          print "aConstraints in CBS: ", aConstraints
          aSolution = self.reconstructPathWithConstraints(ai, initialPositions[ai], goals[ai], aConstraints, pSolution)
          print "aSolution: ", aSolution

          test_constraint = aConstraints[0]
          agentIndex = test_constraint[0]
          solution_with_conflict = aSolution[agentIndex]
          print "solution_with_conflict: ", solution_with_conflict



          aSolution_withCords = self.createPathWithCords(aSolution, goals)
          #print "aSolution_withCords: ", aSolution_withCords
          agent_path = aSolution[agentIndex]
          print "agent_path: ", agent_path
          
          agentPathInCords = self.convertPathWithDirectionsToCords(agentIndex, agent_path, goals[agentIndex])
          #doesPathRequireBackChaining(self, agent, path, constraint):
          prune = self.doesPathRequireBackChaining(agentIndex, agentPathInCords, aConstraints[0])
          if ( prune == True):
            print "YES"
            test_aSolution = self.singleAgentRecoveryMethod(agentIndex, agentPathInCords, aConstraints, goals[agentIndex])
            print "test_aSolution: ", test_aSolution
          if (prune == False):
            #test_aSolution = 
            print "NO"
            test_aSolution = [agent_path, aConstraints]
            print "test_aSolution: ", test_aSolution
          #aSolution[agentIndex] = test_aSolution['solution'] 
          singleAgentSolution = test_aSolution[0]
          back_chain_constraints = test_aSolution[1]
          aSolution[agentIndex] = singleAgentSolution
          print "aSolution after singleAgentRecoveryMethod: ", aSolution

          if self.hasNoConflicts(aSolution, goals) == True:
            print "CBS SUCCESS!!!"
            #print "solution: ", p.data['solution']
            return aSolution

          #new_aData = self.enterRecoveryMethod(agentIndex, aSolution_withCords, aConstraints, goals)
          test_path_for_cost = aSolution[0]
          for index in aConstraints:
            if index not in back_chain_constraints:
              back_chain_constraints.append(index)
          print "aConstraints: ", aConstraints, "back_chain_constraints: ", back_chain_constraints, "pConstraints: ", pConstraints 



          new_aData = {'solution': aSolution, 'cost': len(test_path_for_cost), 'constraints':back_chain_constraints}
          print "new_aData for real maybe?: ", new_aData
          new_aSolution = new_aData['solution']
          #print "new_aSolution: ", new_aSolution
          #print "len(new_aSolution): ", len(new_aSolution)
          #singlePath = new_aSolution[0]
          new_aCost = new_aData['cost']
          new_aConstraint = new_aData['constraints']
          #print "new_aConstraint: ", new_aConstraint
          #print "YESSSS new_aData: ", new_aData

          #aData = {'cost':new_aCost, 'constraints':new_aConstraint, 'solution':new_aSolution} 
          #print "aData: ", aData
          print "jj = ", jj
          if jj == 1:
            p.insertLeft(new_aData)
            p.getLeftChild().setRootVal(new_aData)
            print "p's left child: ", p.getLeftChild().getRootVal()
          if jj == 2:
            
            print"getting root value from CT: ", CT.getRootVal()
            print"getting root value from p: ", p.getRootVal()
            p.insertRight(new_aData)
            p.getRightChild().setRootVal(new_aData)
            print "p's right child: ", p.getRightChild().getRootVal()
          jj+=1

          print"getting root value from CT: ", CT.getRootVal()
          #print(p.getLeftChild().getRootVal())
          #a = CT.insert(root, new_aData)
          #print "PRINTING THE TREE: "
          #CT.printTree(a)
          #print "setting root value for p:"
          p.setRootVal(new_aData)
          print"getting root value from p: ", p.getRootVal()
          #print "p's left child: ", p.getLeftChild().getRootVal()
          #print "p's right child: ", p.getRightChild().getRootVal()

          fringe.push(p, new_aCost)
          print "pSolution: ", pSolution
          print "aSolution: ", new_aSolution
          print "pConstraints: ", pConstraints
          print "aConstraints: ", new_aConstraint
          print "pCost: ", pCost
          print "aCost: ", new_aCost
          
        m+=1
    return []
        
 

    #python mapf_pacman.py --frameTime 2
  def MAPFinder( self):

    initialPositions = [] 
    goalPositions = []    
    paths = []           
    

    
    for i in range(len(self.agents)):
      initialPositions.append(self.state.data.agentStates[ i ].getPosition())
      paths.append([])

    print "pacman start position: ", self.state.data.agentStates[ 0 ].getPosition()
    print "ghost1 start position: ", self.state.data.agentStates[ 1 ].getPosition()
    print "ghost2 start position: ", self.state.data.agentStates[ 2 ].getPosition()

    # making variables for the agent's goalPostions instead of using hard-coded values 
    goalX1 = 2
    goalX2 = 1
    goalX3 = 1
    goalY1 = 2
    goalY2 = 2 
    goalY3 = 1

    #goalPositions[ 0 ] = [(18, 1)] -- pacman's goal
    #goalPositions[ 1 ] = [(1,9)]   -- ghost # 1's goal
    #goalPositions[ 2 ] = [(18,9)]  -- ghost #2's goal
    goalPositions = [(goalX1 , goalY1), (goalX2 , goalY2), (goalX3 , goalY3)] 
    
    
    #print "walls: \n", self.state.data.layout.walls
    #make sure self.state.data.layout.walls[goalX][goalY] == False
    if self.state.data.layout.walls[goalX1][goalY1] == True:
       return
    if self.state.data.layout.walls[goalX2][goalY2] == True:
       return
    if self.state.data.layout.walls[goalX3][goalY3] == True:
       return

    #agentIndex =  depth % state.getNumAgents()
    #print "Initial Positions: ", initialPositions 

    '''
    paths = self.aStarPathFinding(initialPositions, goalPositions)
    print "A* finished: ", paths

    agentPaths = [] 
    # pacman's path = agentPaths[0]
    agentPaths.append(paths['pacman'])
    # ghost1's path = agentPaths[1]
    agentPaths.append(paths['ghost1'])
    # ghost2's path = agentPaths[2]
    agentPaths.append(paths['ghost2'])
    '''

    
    agentPaths = []
    for i in range(len(self.agents)):
      path = self.singleAgentAstarSearch(initialPositions[i], goalPositions[i])
      agentPaths.append(path)
      #self.agents[ i ].setPathPlan( agentPaths[ i ])
    print"look here: ", agentPaths
    solution = self.makePathsTheSameLength(agentPaths)
    print "solution: ", solution
    new_path = self.conflictBasedSearch(solution, goalPositions)
    print "new_path: ", new_path

    for i in range(len(self.agents)):
      #path = self.singleAgentAstarSearch(initialPositions[i], goalPositions[i])
      #agentPaths.append(path)
      self.agents[ i ].setPathPlan( new_path[ i ])

    
    #that's it; 
    return
    ############################################################################################################################################################################
    #                                   ************** HERE IS WHERE MY CODE ENDS **************
    ############################################################################################################################################################################









  def run( self ):
    """
    Main control loop for game play.
    """
    self.display.initialize(self.state.data)
    self.numMoves = 0

    self.MAPFinder()
    
    ###self.display.initialize(self.state.makeObservation(1).data)
    # inform learning agents of the game start
    for i in range(len(self.agents)):
      agent = self.agents[i]
      if not agent:
        # this is a null agent, meaning it failed to load
        # the other team wins
        self._agentCrash(i, quiet=True)
        return
      if ("registerInitialState" in dir(agent)):
        self.mute()
        if self.catchExceptions:
          try:
            timed_func = TimeoutFunction(agent.registerInitialState, int(self.rules.getMaxStartupTime(i)))
            try:
              start_time = time.time()
              timed_func(self.state.deepCopy())
              time_taken = time.time() - start_time
              self.totalAgentTimes[i] += time_taken
            except TimeoutFunctionException:
              print "Agent %d ran out of time on startup!" % i
              self.unmute()
              self.agentTimeout = True
              self._agentCrash(i, quiet=True)
              return
          except Exception,data:
            self.unmute()
            self._agentCrash(i, quiet=True)
            return
        else:
          agent.registerInitialState(self.state.deepCopy())
        ## TODO: could this exceed the total time
        self.unmute()

    agentIndex = self.startingIndex
    numAgents = len( self.agents )

    while not self.gameOver:
      # Fetch the next agent
      agent = self.agents[agentIndex]
      move_time = 0
      skip_action = False
      # Generate an observation of the state
      if 'observationFunction' in dir( agent ):
        self.mute()
        if self.catchExceptions:
          try:
            timed_func = TimeoutFunction(agent.observationFunction, int(self.rules.getMoveTimeout(agentIndex)))
            try:
              start_time = time.time()
              observation = timed_func(self.state.deepCopy())
            except TimeoutFunctionException:
              skip_action = True
            move_time += time.time() - start_time
            self.unmute()
          except Exception,data:
            self.unmute()
            self._agentCrash(agentIndex, quiet=True)
            return
        else:
          observation = agent.observationFunction(self.state.deepCopy())
        self.unmute()
      else:
        observation = self.state.deepCopy()

      # Solicit an action
      action = None
      self.mute()
      if self.catchExceptions:
        try:
          timed_func = TimeoutFunction(agent.getAction, int(self.rules.getMoveTimeout(agentIndex)) - int(move_time))
          try:
            start_time = time.time()
            if skip_action:
              raise TimeoutFunctionException()
            action = timed_func( observation )
          except TimeoutFunctionException:
            print "Agent %d timed out on a single move!" % agentIndex
            self.agentTimeout = True
            self.unmute()
            self._agentCrash(agentIndex, quiet=True)
            return

          move_time += time.time() - start_time

          if move_time > self.rules.getMoveWarningTime(agentIndex):
            self.totalAgentTimeWarnings[agentIndex] += 1
            print "Agent %d took too long to make a move! This is warning %d" % (agentIndex, self.totalAgentTimeWarnings[agentIndex])
            if self.totalAgentTimeWarnings[agentIndex] > self.rules.getMaxTimeWarnings(agentIndex):
              print "Agent %d exceeded the maximum number of warnings: %d" % (agentIndex, self.totalAgentTimeWarnings[agentIndex])
              self.agentTimeout = True
              self.unmute()
              self._agentCrash(agentIndex, quiet=True)

          self.totalAgentTimes[agentIndex] += move_time
          #print "Agent: %d, time: %f, total: %f" % (agentIndex, move_time, self.totalAgentTimes[agentIndex])
          if self.totalAgentTimes[agentIndex] > self.rules.getMaxTotalTime(agentIndex):
            print "Agent %d ran out of time! (time: %1.2f)" % (agentIndex, self.totalAgentTimes[agentIndex])
            self.agentTimeout = True
            self.unmute()
            self._agentCrash(agentIndex, quiet=True)
            return
          self.unmute()
        except Exception,data:
          self.unmute()
          self._agentCrash(agentIndex)
          return
      else:
        action = agent.getAction(observation)
      self.unmute()

      # Execute the action
      self.moveHistory.append( (agentIndex, action) )
      if self.catchExceptions:
        try:
          self.state = self.state.generateSuccessor( agentIndex, action )
        except Exception,data:
          self._agentCrash(agentIndex)
          return
      else:
        self.state = self.state.generateSuccessor( agentIndex, action )

      # Change the display
      self.display.update( self.state.data )
      ###idx = agentIndex - agentIndex % 2 + 1
      ###self.display.update( self.state.makeObservation(idx).data )

      # Allow for game specific conditions (winning, losing, etc.)
      self.rules.process(self.state, self)
      # Track progress
      if agentIndex == numAgents + 1: self.numMoves += 1
      # Next agent
      agentIndex = ( agentIndex + 1 ) % numAgents

      if _BOINC_ENABLED:
        boinc.set_fraction_done(self.getProgress())

    # inform a learning agent of the game result
    for agent in self.agents:
      if "final" in dir( agent ) :
        try:
          self.mute()
          agent.final( self.state )
          self.unmute()
        except Exception,data:
          if not self.catchExceptions: raise
          self.unmute()
          print "Exception",data
          self._agentCrash(agent.index)
          return
    self.display.finish()




