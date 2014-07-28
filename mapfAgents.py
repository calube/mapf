# keyboardAgents.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import Agent
from game import Directions
import random

class MAPFAgent(Agent):
  """
  An agent that executes a planned path
  """

  def __init__( self, index = 0 ):
    
    self.lastMove = Directions.STOP
    self.index = index
    self.internalCounter = 0
    self.path = []

  def setPathPlan(self, path ):
    self.path = path
    
  def getAction( self, state):
    if self.internalCounter < len(self.path):
      move = self.path[ self.internalCounter ]
    else:
      move = Directions.STOP
    legal = state.getLegalActions(self.index)
    if move not in legal:
      print "ILLEGAL MAPF move: ", move
      move = random.choice(legal)
    self.internalCounter += 1
    return move
