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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action: object) -> float:
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
        foodlist = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # foodfunction
        if foodlist:
            fooddist = min(util.manhattanDistance(newPos, f) for f in foodlist)
            foodscore = 1 / (fooddist + 1)
        else:
            foodscore = 0

        # ghostfunction
        ghostdists = [util.manhattanDistance(newPos, g.getPosition()) for g in newGhostStates]
        minghostdist = min(ghostdists)
        if ghostdists and min(ghostdists) < 2:
            return -float(-100)
        if max(newScaredTimes) > minghostdist:
            ghostscore = 1 / minghostdist
        else:
            ghostscore = 0

        return successorGameState.getScore() + foodscore + 100 * ghostscore


def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        def dfs(gamestate, depth: int, agent):
            actions = gamestate.getLegalActions(agent)
            if depth == 0 or gamestate.isWin() or gamestate.isLose() or not actions:
                return self.evaluationFunction(gamestate), None

            numagents = gamestate.getNumAgents()
            nextagent = (agent + 1) % numagents
            nextdepth = depth - 1 if nextagent == 0 else depth

            bestscore = float('-inf') if agent == 0 else float('inf')
            bestaction = None

            for action in actions:
                successor = gamestate.generateSuccessor(agent, action)
                curr_score, _ = dfs(successor, nextdepth, nextagent)
                if agent == 0 and curr_score > bestscore: # agent0 = pacman = maximizer
                    bestscore, bestaction = curr_score, action
                elif agent != 0 and curr_score < bestscore: # ghosts = minimizer
                    bestscore, bestaction = curr_score, action

            return bestscore, bestaction

        _, action = dfs(gameState, self.depth, 0)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def dfs(gamestate, depth: int, agent, alpha: int, beta: int):
            actions = gamestate.getLegalActions(agent)
            if depth == 0 or gamestate.isWin() or gamestate.isLose() or not actions:
                return self.evaluationFunction(gamestate), None

            numagents = gamestate.getNumAgents()
            nextagent = (agent + 1) % numagents
            nextdepth = depth - 1 if nextagent == 0 else depth

            if agent == 0:
                bestscore, bestaction = float('-inf'), None
                for action in actions:
                    successor = gamestate.generateSuccessor(agent, action)
                    curr_score, _ = dfs(successor, nextdepth, nextagent, alpha, beta)
                    if curr_score > bestscore:
                        bestscore, bestaction = curr_score, action
                    alpha = max(alpha, bestscore)
                    if bestscore > beta:
                        break
                return bestscore, bestaction
            else:
                bestscore, bestaction = float('inf'), None
                for action in actions:
                    successor = gamestate.generateSuccessor(agent, action)
                    curr_score, _ = dfs(successor, nextdepth, nextagent, alpha, beta)
                    if curr_score < bestscore:
                        bestscore, bestaction = curr_score, action
                    beta = min(beta, bestscore)
                    if bestscore < alpha:
                        break
                return bestscore, bestaction

        _, action = dfs(gameState, self.depth, 0, float('-inf'), float('inf'))
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        def max_value(state: GameState, depth):
            val = float('-inf')
            actions = state.getLegalActions(0)
            if not actions:
                return self.evaluationFunction(state)
            for action in actions:
                val = max(val, value(state.generateSuccessor(0, action), 1, depth))
            return val

        def expect_value(state: GameState, agent, depth):
            actions = state.getLegalActions(agent)
            if not actions:
                return self.evaluationFunction(state)
            expect_val = 0
            p = 1 / len(actions)
            nextagent = (agent + 1)% state.getNumAgents()
            nextdepth = depth + 1 if nextagent == 0 else depth

            for action in actions:
                successor = state.generateSuccessor(agent, action)
                expect_val += p * value(successor, nextagent, nextdepth)

            return expect_val

        def value(state, agent, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            if agent == 0:
                return max_value(state, depth)
            else:
                return expect_value(state, agent, depth)

        bestscore, bestaction = float('-inf'), None
        actions = gameState.getLegalActions(0)
        if not actions:
            return self.evaluationFunction(gameState)
        for action in actions:
            curr_score = value(gameState.generateSuccessor(0, action), 1, 0)
            if curr_score > bestscore:
                bestscore, bestaction = curr_score, action

        return bestaction

def betterEvaluationFunction(currentGameState: GameState) -> int:
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    1. In foodfunction:
        1) Encourage less remaining food.
        2) Encourage lower distance to the closest food so that pacman can get one more easily.
    2. In capsulefunction:
        1) Encourage fewer remaining capsules (More weight than remaining food as eating a capsule gets more reward).
    3. In ghostfunction:
        1) If a ghost is scared and pacman seems able to eat it, then encourage.
        2) If a ghost isn't scared and is rather close to pacman, punish severely to avoid dying.
    """

    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()

    score = currentGameState.getScore()

    # foodfunction
    score -= 10 * len(foodList)
    if foodList:
        min_food_dist = min(util.manhattanDistance(pacmanPos, food) for food in foodList)
        score -= 1.5 * min_food_dist

    # capsulefunction
    score -= 20 * len(capsules)

    # ghostfunction
    ghost_score = 0
    for ghost in ghostStates:
        dist = util.manhattanDistance(pacmanPos, ghost.getPosition())

        if ghost.scaredTimer > 0:
            if ghost.scaredTimer > dist:
                ghost_score += 200 / (dist + 1)
        else:
            if dist < 2:
                ghost_score -= 1000
    score += ghost_score

    return score

# Abbreviation
better = betterEvaluationFunction