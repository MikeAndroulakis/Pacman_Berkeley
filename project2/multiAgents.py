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
	#print "scores:",scores
        bestScore = max(scores)
	#print "bestscore:",bestScore
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
	#print "bestIndices:",bestIndices
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
	#print "choseindex:",chosenIndex

        "Add more of your code here if you want to"
	#print "legalmoves:",legalMoves[chosenIndex]
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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
	ghostpositions=successorGameState.getGhostPositions()#pinakas me ta positions twn teratwn
	walls = successorGameState.getWalls()#metabliti poy epistrefei ton arithmo twn toixwn wste na ypologisoyme grammes kai sthles
	columns=walls.width-2#arithmos sthlwn
	lines=len(newFood[0])#arithmos grammwn
	value=successorGameState.getScore()#synarthsh poy aksiologei thn katastash poy eperxetai to pacman
	mindistancefromghost=float("inf")#arxikopoihsh se apeiro gia th prwth epanalipsi
	mindistancefromfood=float("inf")#arxikopoihsh se apeiro gia th prwth epanalipsi
	if action=='Stop':#oi staseis toy pacman de boithane opote apotrepontai
	    return (-(float("inf")))
	for ghostposition in ghostpositions:#gia oles tis theseis twn teratwn
	    distancefromghost=util.manhattanDistance(ghostposition,newPos)#ypologizetai h apostash aytoy toy teratos kai toy pacman
	    if distancefromghost<=1:#an exei erthei poly konta sto pacman h kinhsh apotrepetai
		return (-(float("inf")))
	    if distancefromghost<mindistancefromghost:#epilegetai h mikroterh apostash apo ola ta terata
		mindistancefromghost=distancefromghost
	ghostvalue=5.0/float(mindistancefromghost)#h "aksia" twn teratwn eksartatai apo th kontinoterh apostash teratos-pacman
	if newFood[newPos[0]][newPos[1]]==True:#an to epomeno position peftei panw se food epestrepse poly megalh timh
	    value=100
	    return value
	for y in range(1,columns+1): #gia ta columns
	    for x in range(1,lines-1):#gia tis grammes
		if newFood[y][x]==True:#an yparxei food ekei
		    fooddistance=util.manhattanDistance(newPos,(y,x))#ypologizetai h apostash toy faghtoy aytoy kai toy pacman
		    if fooddistance<mindistancefromfood:#briskoyme to faghto me th kontinoterh apostash
			mindistancefromfood=fooddistance
	foodvalue=5.0/float(mindistancefromfood)#h "aksia" twn foods eksartatai apo th mikroterh apostash food-pacman
	v=foodvalue-ghostvalue+value#epistrefetai h aksia ayths ths kinisis ws syndiasmos twn 3 timwn
	return v

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
	#print self.depth
	#print self.evaluationFunction(gameState)
	#gameState.generateSuccessor
	#print gameState.getLegalActions(1)
	#print self.numberofagents
	self.numberofagents=gameState.getNumAgents()#arithmos twn agents(pacman+ghosts)
	maxvalue=self.MAXVALUE(gameState,0)#kaleitai h MAXVALUE me depth 0(h MAXVALUE yparxei mono gia ton agent 0(pacman))
	return maxvalue[1]

    def MAXVALUE(self,gameState,depth):
	v=-float(("inf"))
	actions=gameState.getLegalActions(0)#pinakas twn kinisewn toy pacman
	if gameState.isWin() or gameState.isLose() or len(actions)==0 or depth==self.depth:#an o pacman kerdizei me ayth ti kinisi
	    return (self.evaluationFunction(gameState),None)#h an o pacman peftei panw se teras h an ftasame sto epithimito bathos 
	for action in actions:#gia ola ta actions
	    temp=self.MINVALUE(1,gameState.generateSuccessor(0,action),depth)#kalountai oi MINVALUE gia to teras 1
	    temp=temp[0]#briskoyme thn value ayths ths kinhshs(kathws epistrefetai kai timh kai action)
	    if temp>v:#an h timh poy epistrafike einai megalyterh apo th proigoymenh apothikeyoyme aythn
		v=temp
		bestact=action#apothikeysh toy kalyteroy action
	return (v,bestact)

    def MINVALUE(self,agent,gameState,depth):
	actions=gameState.getLegalActions(agent)
	if len(actions)==0:#an den yparxoyn kiniseis gia to teras tote epistrefetai h evaluation
	    return (self.evaluationFunction(gameState),None)
	v=float("inf")
	if agent==self.numberofagents-1:#an ftasame sto teleytaio teras
	    for action in actions:#gia ola ta action toy teratos
		temp=self.MAXVALUE(gameState.generateSuccessor(agent,action),depth+1)#kaleitai h MAXVALUE gia to pacman kai ayksanetai to bathos
		temp=temp[0]
		if temp<v:
		    v=temp
		    bestact=action
	    return (v,bestact)
	else:#alliws an den eimaste sto teleytaio teras
	    for action in actions:
		temp=self.MINVALUE(agent+1,gameState.generateSuccessor(agent,action),depth)#kaleitai pali h MINVALUE gia to epomeno ghost
		temp=temp[0]
		if temp<v:
		    v=temp
		    bestact=action
	    return (v,bestact)
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.numberofagents=gameState.getNumAgents()#arithmos twn agents
	maxvalue=self.MAXVALUE(gameState,0,bestmax=-(float("inf")),bestmin=float("inf"))#kaleitai h MAXVALUE gia ton pacman
	return maxvalue[1]

    def MAXVALUE(self,gameState,depth,bestmax,bestmin):
	v=-float(("inf"))
	actions=gameState.getLegalActions(0)
	bestact=None
	if gameState.isWin() or gameState.isLose() or len(actions)==0 or depth==self.depth:
	    return (self.evaluationFunction(gameState),None)
	for action in actions:
	    if action!='Stop':#na mh kanei staseis
	        temp=self.MINVALUE(1,gameState.generateSuccessor(0,action),depth,bestmax,bestmin)#kaleitai h MINVALUE gia to ghost 1
	        temp=temp[0]
	        if temp>v:
		    v=temp
		    bestact=action
	        if v>bestmin:#an h timh einai megalyterh apo to min tote den exoyn nohma oi ypoloipoi komvoi kai ginetai kladema
		    return (v,bestact)
	        bestmax=max(bestmax,v)#briskoyme th megalyterh timh toy MAX 
	if bestact==None:#an den yparxei kinisi tote stamataei gia ligo o pacman
	    return (v,'Stop')
	return (v,bestact)

    def MINVALUE(self,agent,gameState,depth,bestmax,bestmin):
	actions=gameState.getLegalActions(agent)
	if len(actions)==0:#an den yparxoyn kiniseis epistrefoyme thn evaluation
	    return (self.evaluationFunction(gameState),None)
	v=float("inf")
	if agent==self.numberofagents-1:#an ftasame sto teleytaio teras
	    bestact=None
	    for action in actions:
		temp=self.MAXVALUE(gameState.generateSuccessor(agent,action),depth+1,bestmax,bestmin)#kaleitai h MAXVALUE gia pacman kai ayksanetai to bathos
		temp=temp[0]#apothikeysh ths timis tis kinisis
		if temp<v:
		    v=temp#briskoyme th kalyterh timh
		    bestact=action#apothikeyoyme th kalyterh kinhsh
		if v<bestmax:#an h timi ths MAXVALUE einai mikroterh apo to yparxon max den exei nohma h synexeia kai ginetai kladema
		    return (v,bestact)
		bestmin=min(bestmin,v)#briskoyme th mikroterh timh
	    if bestact==None:
		return (v,'Stop')
	    return (v,bestact)
	else:#an den eimaste sto teleytaio ghost ksanakaloyme thn MINVALUE gia to epomeno ghost
	    bestact=None
	    for action in actions:
		temp=self.MINVALUE(agent+1,gameState.generateSuccessor(agent,action),depth,bestmax,bestmin)
		temp=temp[0]
		if temp<v:
		    v=temp
		    bestact=action
		if v<bestmax:#an h timh ths MINVALUE einai mikroterh apo to yparxon max oi epomenoi komvoi den exoyn nohma kai kladeyontai
		    return (v,bestact)
		bestmin=min(bestmin,v)#briskoyme ti mikroteri timh
	    if bestact==None:
		return (v,'Stop')
	    return (v,bestact)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.numberofagents=gameState.getNumAgents()#arithmos agents
        actions=gameState.getLegalActions(0)
        v=-(float("inf"))
        for action in actions:
	    temp=self.expectedvalueforghosts(gameState.generateSuccessor(0,action), 1,0)#kaleitai h synarthsh gia bathos 0
	    if temp>v:#an brethike kalyterh timh apothikeyetai h action
		v=temp
		bestact=action
	return bestact

    def expectedvalueforghosts(self,gameState,agent,depth):#synarthsh gia pithanotita twn ghosts
	if gameState.isWin() or gameState.isLose():
	    return self.evaluationFunction(gameState)
	actions=gameState.getLegalActions(agent)
	Sum=0#metabliti poy krataei to athroisma twn timwn wste na ypologisoyme tis pithanotites
        for action in actions:
            if (agent==self.numberofagents-1):#an ftasame sto teleytaio ghost,ayksanoyme to depth kai kaloyme th synarthsh gia pacman
		Sum+=self.maxforpacman(gameState.generateSuccessor(agent,action),0,depth+1)#ypologismos athroismatos
            else:#alliws
		Sum+=self.expectedvalueforghosts(gameState.generateSuccessor(agent,action),agent+1,depth)#kaleitai h synarthsh gia to epomeno ghost
	chance=float(Sum)/float(len(actions))#ypologizoyme th pithanotita aytoy toy teratos me bash to athroisma twn timwn twn kinisewn kai ton arithmo twn kinisewn
	return chance
    def maxforpacman(self,gameState,agent,depth):#synarthsh gia pithanotita toy pacman
	if gameState.isWin() or gameState.isLose() or depth==self.depth:
	    return self.evaluationFunction(gameState)
	v=-(float("inf"))
	actions=gameState.getLegalActions(0)
	for action in actions:
	    temp=self.expectedvalueforghosts(gameState.generateSuccessor(0, action), 1, depth)#kaleitai h synarthsh gia to ghost 1
	    if temp>v:#briskoyme th kaliteri timh gia tis kiniseis toy pacman
		v=temp
	return v
	
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    Pos=currentGameState.getPacmanPosition()
    newFood=currentGameState.getFood()
    newGhostStates=currentGameState.getGhostStates()
    newScaredTimes=[ghostState.scaredTimer for ghostState in newGhostStates]
    columns=currentGameState.getWalls().width-2#apothikeyoyme ton arithmo twn sthlwn gia to loop twn foods
    lines=len(newFood[0])#apothikeysh grammwn
    mindistancefromfood=float("inf")
    value=currentGameState.getScore()#synarthsh poy aksiologei thn yparxoysa katastash
    for y in range(1,columns+1): #gia ta columns
	for x in range(1,lines-1):#gia tis grammes
       	    if newFood[y][x]==True:
		fooddistance=util.manhattanDistance(Pos,(y,x))#ypologismos apostashs food-pacman
		if fooddistance<mindistancefromfood:#eyresh elaxistis apostashs food-pacman
		    mindistancefromfood=fooddistance	    
    valueforfood=5.0/float(mindistancefromfood)#h aksia toy food me dokimes timwn
    valueforghosts=0
    for i in range(len(newGhostStates)):#gia ola ta terata
	ghostpos=newGhostStates[0].getPosition()#h thesi aytoy toy teratos
	distancefromghost=manhattanDistance(Pos,ghostpos)#apostash toy pacman kai toy ghost
	if distancefromghost>0:#an den einai sto idio tetragwno
	    if newGhostStates[i].scaredTimer>0:#an ayto to ghost einai fobismeno
	        valueforghosts+=50.0/float(distancefromghost)#ayksise kata poly to synoliko value ths aksias twn ghosts
	    else:#alliws
	        valueforghosts-=5.0/float(distancefromghost)#h aksia twn ghosts einai analogh ths apostashs apto pacman
    v=valueforfood+valueforghosts+value#h aksia ayths tis kinhshs ypologizetai synarthsei aytwn twn timwn
    return v
	

# Abbreviation
better = betterEvaluationFunction
