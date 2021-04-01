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
    "*** YOUR CODE HERE ***"
    start=problem.getStartState() #apothikeyetai h arxh toy project
    if problem.name=='CornersProblem':#ean briskomaste sto cornersproblem to start einai alliwtiko kathws kathws ektos apo
	start=problem.start           #tis syntetagmenes toy pacman twra exoyme kai ena adeio tuple
    graph={}#grafos gia ton ypologismo toy monopatioy
    frontier=util.Stack()#orizoume mia stoiva
    frontier.push(start)#isagetai to start sto stack
    exploredset=[]#lista poy apothikeyontai oi komboi poy exoyme episkefthei
    path=[]#edw apothikeyetai to path ths lisis
    temp2=0
    if(problem.isGoalState(start)):#an h arxh einai goal state tote termatismos
	print("The start state is the goal")
	return 1
    while 1:
	if(frontier.isEmpty()):#elegxos an to frontier einai adeio
	    print("Failure")
	    return -1
        node=frontier.pop()
	exploredset.insert(len(exploredset),node)#o komvos poy bghke apo to frontier isagetai sth lista twn explored
	for action in problem.getSuccessors(node):#gia kathe energeia apo tis epitreptes poy mporoyn na ginoyn
	    if action[0] not in frontier.list and action[0] not in exploredset:#an h energeia den einai stis listes
		graph[action[0]]=node#apothikeyetai o proigoymenos komvos gia thn metepeita anadromh
		if(problem.isGoalState(action[0])):#ean brikame to stoxo
		    if problem.name=='CornersProblem':#an to problhma einai corners tote apothikeyontai alles metablites
			child=action[0][0]            #kathws sto problhma ayto exoyme diaforetiki domh gia ta nodes
			parent=node[0]#o parent isoutai me tis syntetagmenes (x,y) toy node
			temp2=node#edw apothikeyetai olh h domh toy node gia metepeita anadromh mesw toy graph
		    else:#alliws an briskomaste se opoiodipote allo problhma
		        child=action[0]#sto child apothikeyetai oi syntetagmenes toy action[0]
		        parent=node#sto node apothikeyontai oi syntetagmenes toy proigoymenoy node apo to action[0]
		    while 1:#mexri na ftasoyme sthn arxh toy pacman
			if child[0]==parent[0] and child[1]<parent[1]:#an h epomenh thesi einai pio panw tote
			    path.insert(0,'South')#apothikeyetai sto path to South
			elif child[0]==parent[0] and child[1]>parent[1]:#an h epomenh thesi einai pio katw tote
			    path.insert(0,'North')#apothikeyetai sto path to North
			elif child[0]<parent[0] and child[1]==parent[1]:#an h epomenh thesi einai pio deksia tote
			    path.insert(0,'West')#apothikeyetai sto path West
			elif child[0]>parent[0] and child[1]==parent[1]:#an h epomenh thesi einai pio aristera tote
			    path.insert(0,'East')#apothikeyetai sto path East
			if parent==start:#ean me thn anadromh ftasame sthn arxh tote stamatame
			    break
			if temp2==start:#gia to corners problem ean me thn anadromh ftasame sthn arxh stamatame
			    break;
			if problem.name=='CornersProblem':#an briskomaste sto cornersproblem
			    child=temp2
			    child=child[0]#sto child apothikeyontai mono oi syntetagmenes
			    temp2=graph[temp2]#edw ginetai h anadromh kai sto temp2 yparxoyn oi syntetagmenes alla kai h lista 
			    parent=temp2[0]   #me ta corners
			else:#an den eimaste sto cornersproblem
			    previousparent=parent
			    parent=graph[parent]#kai o pateras ginetai h epomenh energeia
			    child=previousparent#o pateras ginetai to paidi 
		    return path
		frontier.push(action[0])
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    start=problem.getStartState()#apothikeyetai h arxh toy project
    if problem.name=='CornersProblem':#ean briskomaste sto cornersproblem to start einai alliwtiko kathws kathws ektos apo
	start=problem.start#tis syntetagmenes toy pacman twra exoyme kai ena adeio tuple
    graph={}#grafos gia ton ypologismo toy monopatioy
    frontier=util.Queue()#orisoyme mia oyra
    frontier.push(start)#isagetai to start sto stack
    exploredset=[]#lista poy apothikeyontai oi komboi poy exoyme episkefthei
    path=[]#edw apothikeyetai to path ths lisis
    temp2=0
    if(problem.isGoalState(start)):#an h arxh einai goal state tote termatismos
	print("The start state is the goal")
	return 1
    while 1:
	if(frontier.isEmpty()):#elegxos an to frontier einai adeio
	    print("Failure")
	    return -1
        node=frontier.pop()
	exploredset.insert(len(exploredset),node)#o komvos poy bghke apo to frontier isagetai sth lista twn explored
	for action in problem.getSuccessors(node):#gia kathe energeia apo tis epitreptes poy mporoyn na ginoyn
	    if action[0] not in frontier.list and action[0] not in exploredset:#an h energeia den einai stis listes
		graph[action[0]]=node#apothikeyetai o proigoymenos komvos gia thn metepeita anadromh
		if(problem.isGoalState(action[0])):#an to problhma einai corners tote apothikeyontai alles metablites
		    if problem.name=='CornersProblem':#an to problhma einai corners tote apothikeyontai alles metablites
			child=action[0][0]            #kathws sto problhma ayto exoyme diaforetiki domh gia ta nodes
			parent=node[0]#o parent isoutai me tis syntetagmenes (x,y) toy node
			temp2=node#edw apothikeyetai olh h domh toy node gia metepeita anadromh mesw toy graph
		    else:#alliws an briskomaste se opoiodipote allo problhma
		        child=action[0]#sto child apothikeyetai oi syntetagmenes toy action[0]
		        parent=node#sto node apothikeyontai oi syntetagmenes toy proigoymenoy node apo to action[0]
		    while 1:
			if child[0]==parent[0] and child[1]<parent[1]:
			    path.insert(0,'South')
			elif child[0]==parent[0] and child[1]>parent[1]:
			    path.insert(0,'North')
			elif child[0]<parent[0] and child[1]==parent[1]:
			    path.insert(0,'West')
			elif child[0]>parent[0] and child[1]==parent[1]:
			    path.insert(0,'East')
			if parent==start:
			    break
			if temp2==start:#gia to corners problem
			    break;
			if problem.name=='CornersProblem':#an briskomaste sto cornersproblem
			    child=temp2
			    child=child[0]#sto child apothikeyontai mono oi syntetagmenes
			    temp2=graph[temp2]#edw ginetai h anadromh kai sto temp2 yparxoyn oi syntetagmenes alla kai h lista 
			    parent=temp2[0]   #me ta corners
			else:#an den eimaste sto cornersproblem
			    previousparent=parent
			    parent=graph[parent]#kai o pateras ginetai h epomenh energeia
			    child=previousparent#o pateras ginetai to paidi 
		    return path
		frontier.push(action[0])
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    start=problem.getStartState()#apothikeyetai h arxh toy project
    graph={}#grafos gia ton ypologismo toy monopatioy
    frontier=util.PriorityQueue()#orizoyme mia oyra proteraiotitas
    frontier.push(start,0)
    exploredset=[]
    path=[]
    if(problem.isGoalState(start)):
	print("The start state is the goal")
	return 1
    while 1:
	if(frontier.isEmpty()):
	    print("Failure")
	    return -1
        node=frontier.pop()
	exploredset.insert(len(exploredset),node)
	for action in problem.getSuccessors(node):#gia kathe epitrepto action
	    if action[0] not in frontier.heap and action[0] not in exploredset:#an to action ayto den yparxei stis listes
		graph[action[0]]=node#apothikeyetai o proigoymenos komvos gia thn anadromh
		child=action[0]#to paidi isoytai me tis syntetagmenes (x,y) toy action[0]
		parent=node#o pateras isoytai me tis syntetagmenes (x,y) toy node
		while child!=start:#mexri na ftasoyme sthn arxh
		    if child[0]==parent[0] and child[1]<parent[1]:#an h epomenh thesi einai pio panw tote
			path.insert(0,'South')#apothikeyetai sto path to South
		    elif child[0]==parent[0] and child[1]>parent[1]:#an h epomenh thesi einai pio katw tote
			path.insert(0,'North')#apothikeyetai sto path North
		    elif child[0]<parent[0] and child[1]==parent[1]:#an h epomenh thesi einai pio deksia tote
			path.insert(0,'West')#apothikeyetai sto path West
		    elif child[0]>parent[0] and child[1]==parent[1]:#an h epomenh thesi einai pio aristera tote
			path.insert(0,'East')#apothikeyetai sto path East
		    if parent==start:#an ftasame sthn arxh stamatame
			cost=problem.getCostOfActions(path)#apothikeyetai to kostos gia na ftasoyme apo thn arxh se ayto to komvo
			break
		    previousparent=parent
		    parent=graph[parent]#o pateras isoytai me thn epomenh kinhsh symfwna me thn anadromh toy grafoy
		    child=previousparent#to paidi ginetai o proigoymenos pateras
		if(problem.isGoalState(action[0])):
		    return path
		del path[:]#diagrafetai h lista toy path gia thn epomenh epanalipsi
		frontier.push(action[0],cost)#mpainei sthn oyra me kapoio kostos
	    elif action[0] in frontier.heap:#an to action[0] einai hdh sto frontier
		if node[0]==action[0] and node[1]<action[1]:#apothikeyetai sto path h teleytaia kinhsh poy kaname
		    path.insert(0,'South')                  #diladi h kinhsh apo to node sto action[0]
		elif node[0]==action[0] and node[1]>action[1]:
		    path.insert(0,'North')
		elif node[0]<action[0] and node[1]==action[1]:
		    path.insert(0,'West')
		elif node[0]>action[0] and node[1]==action[1]:
		    path.insert(0,'East')
		for element in frontier.heap:#gia kathe element poy yparxei mesa sto heap toy frontier
		    if element==action[0]:#ean brethike to action[0]
			for i in range(0,2):
			    if i==0:#h prwth epanalipsi einai gia to action[0] (opoy proyparxei mesa sth lista toy path h teleytaia
                                    # kinhsh toy pacman wste na mh ginei mpleksimo anadromhs me ton grafo)
			        child=node
			        parent=graph[node]#kai ousiastika einai san na sygkrinoyme to node me to element,alla den einai etsi
						  #kathws exei perastei hdh mesa h teleytaia kinhsh apo to node mexri to action[0]
			    elif i==1:#h deyterh epanalipsi einai gia to element
			        child=element
			        parent=graph[child]#opoy edw kanonika leitoytgei h anadromh toy grafoy
			    while child!=start:#mexri na pame sthn arxh
			        if child[0]==parent[0] and child[1]<parent[1]:
			            path.insert(0,'South')
			        elif child[0]==parent[0] and child[1]>parent[1]:
			    	    path.insert(0,'North')
			        elif child[0]<parent[0] and child[1]==parent[1]:
			            path.insert(0,'West')
			        elif child[0]>parent[0] and child[1]==parent[1]:
			            path.insert(0,'East')
			        if parent==start:#ean ftasame sthn arxh
				    if i==0:#an eimaste sthn epanalipsi toy node
				        cost1=problem.getCostOfActions(path)#apothikeyetai to kostos ths diadromis
				    elif i==1:#an eimaste sthn epanalipsi toy action[0]
					cost2=problem.getCostOfActions(path)#apothikeyetai to kostos ths diadromis toy
				    del path[:]#sbhnetai h lista gia thn epomenh epanalhpsh
			            break
			        previousparent=parent
			        parent=graph[parent]
			        child=previousparent
			if cost1>cost2:#an to kostos einai mikrotero (diladi brethike kalyteri diadromh)
			    frontier.update(action[0],cost2)#allazoyme to action[0] me to kainoyrio kalytero kostos
			    graph[action[0]]=node#apothikeyoyme to katallilo monopati gia th swsth anadromh
		        break
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    graph={}
    start=problem.getStartState()
    if problem.name=='CornersProblem' or problem.name=='FoodSearchProblem':#an exoyme ena apo ayta ta dyo problimata 
	start=problem.start#to start tote ektos apo tis syntetagmenes (x,y) toy pacman exei kai ena adeio tuple
    frontier=util.PriorityQueue()#orizoyme mia oyra proteraiotitas
    frontier.push(start,0)
    exploredset=[]
    path=[]
    g=[]#lista gia tis apostaseis ten komvwn apo thn arxh
    h=[]#lista gia tis eyretikes twn komvwn
    temp2=0
    while 1:
	if(frontier.isEmpty()):
	    print("Failure")
	    return -1
        node=frontier.pop()
	if(problem.isGoalState(node)):
	    if problem.name=='CornersProblem' or problem.name=='FoodSearchProblem':#an exoyme ena apo ta dyo problimata
										   #tote ena node apoteleitai apo tis syntetagmenes
										   #(x,y) alla kai ena adeio tuple gia ayto
										   #kanoyme elegxo
	       child=action[0][0]#apothikeyetai h syntetagmenes (x,y) toy action[0]
	       temp1=action[0]#apothikeyetai olh h plhroforia toy action gia th metepeita anadromh
	       parent=node[0]
	       temp2=node
	    else:
		child=action[0]
		parent=node
	    while 1:
		if child[0]==parent[0] and child[1]<parent[1]:
		    path.insert(0,'South')
		elif child[0]==parent[0] and child[1]>parent[1]:
		    path.insert(0,'North')
		elif child[0]<parent[0] and child[1]==parent[1]:
	            path.insert(0,'West')
		elif child[0]>parent[0] and child[1]==parent[1]:
		    path.insert(0,'East')
		if parent==start or temp2==start:
		    break
	        if problem.name=='CornersProblem' or problem.name=='FoodSearchProblem':#an exoyme ena apo ta 2 problhmata
		    previousparent=temp2
		    child=previousparent
		    child=child[0]
		    temp2=graph[temp2]
		    parent=temp2[0]
		else:
		    previousparent=parent
		    parent=graph[parent]
		    child=previousparent
	    return path
	exploredset.insert(len(exploredset),node)
	i=0
	for action in problem.getSuccessors(node):#gia kathe egkyro action
	    if action[0] not in frontier.heap and action[0] not in exploredset:#poy den einai stis listes
	        graph[action[0]]=node#apothikeyoyme ton proigoymeno gia anadromh
		if problem.name=='CornersProblem' or problem.name=='FoodSearchProblem':
	            child=action[0][0]
	            temp1=action[0]
	            parent=node[0]
	            temp2=node
	        else:
		    child=action[0]
		    parent=node
		while 1:
		    if child[0]==parent[0] and child[1]<parent[1]:
			path.insert(0,'South')
		    elif child[0]==parent[0] and child[1]>parent[1]:
			path.insert(0,'North')
		    elif child[0]<parent[0] and child[1]==parent[1]:
			path.insert(0,'West')
		    elif child[0]>parent[0] and child[1]==parent[1]:
			path.insert(0,'East')
		    if temp2==start or parent==start:#to temp2 einai gia ta corners kai foodsearch problems enw to parent einai
						     #gia thn aplh A*
			g.insert(len(g),problem.getCostOfActions(path))#bazoyme ta kosth sti lista
			h.insert(len(h),heuristic(action[0],problem))#bazoyme tis eyretikes kathe komvoy sth lista
			i+=1#auksanoyme to i gia thn epomenh epanalipsi
			del path[:]#diagrafoyme to path gia thn epomenh epanalipsi
			break
		    if problem.name=='CornersProblem' or problem.name=='FoodSearchProblem':
		        previousparent=temp2[0]
		        child=previousparent
		        temp2=graph[temp2]
		        parent=temp2[0]
		    else:
		        previousparent=parent
		        parent=graph[parent]
		        child=previousparent
	        frontier.push((action[0]),g[i-1]+h[i-1])#kathe action eisagetai sto frontier me thn katallili g kai h
	del g[:]#diagrafetai h g gia thn epomenh epanalispi
	del h[:]#diagrafetai h h gia thn epomenh epanalipsi
    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
