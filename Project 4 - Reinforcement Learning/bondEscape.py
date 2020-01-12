import escapeGame
import numpy as np
import pickle

map = [["x"]*9,\
    ["x","button" ,"x","door"   ,"o"   ,"o","x","button","x"],\
    ["x","enemy"  ,"x","door"   ,"x"   ,"o","x","enemy"  ,"x"],\
    ["x","o"      ,"x","button" ,"x"   ,"o","x","o"      ,"x"],\
    ["x","o"      ,"x","x"      ,"x"   ,"o","o","o"      ,"x"],\
    ["x","o"      ,"o","o"      ,"o"   ,"o","o","o"      ,"x"],\
    ["x","o"      ,"x","o"      ,"o"   ,"o","x","o"      ,"x"],\
    ["x","enemy"  ,"x","o"      ,"o"   ,"o","x","enemy"  ,"x"],\
    ["x","button" ,"x","o"      ,"Bond","o","x","button","x"],\
    ["x","x"      ,"x","x"      ,"exit","x","x","x"      ,"x"]]
 


class QLearningAgent:
    def __init__(self, game, alpha = 0.1, gamma = 0.95, epsilon = 0.05):
        self.game = game
        self.alpha = alpha #learning rate
        self.gamma = gamma #discount rate
        self.epsilon = epsilon #random search rate
        self.Q = {} #dictionary(keys are states) of dictionary (keys are actions, reward is entry)

    def saveQFunction(self, filename = 'default'):
        outputFile = open(filename, 'wb')
        pickle.dump(self.Q, outputFile)
        outputFile.close()

    def loadQFunction(self, filename = 'default'):
        infile = open(filename,'rb')
        self.Q = pickle.load(infile)
        infile.close()

    def train(self, iterations):#epsilon greedy method
        print("Initial game map:")
        self.game.printMap()
        for i in range(iterations): #for loop to play game count
            (stateRecord, actionRecord, rewardRecord) = self.playLearningGame(self.epsilon)
            self.updateQFunction(stateRecord, actionRecord, rewardRecord)
            print("Iteration " + str(i+1) + " of " + str(iterations))
            print("\tTotal reward = " + str(sum(rewardRecord)) + " after " + str(len(rewardRecord)) + " moves")
            print("\tLast move: " + actionRecord[-1])
            print("Final Map: ")
            self.game.printMap()

    def updateQFunction(self, stateRecord, actionRecord, rewardRecord):
        for N in range(len(stateRecord)-1,-1,-1):#Reverse order of N accounts
            #Learn current value
            self.Q[stateRecord[N]][actionRecord[N]] = (1-self.alpha)*self.Q[stateRecord[N]][actionRecord[N]] + \
                self.alpha*rewardRecord[N]
            #Learn future value
            if N != (len(stateRecord)-1):
                bestReward = float("-inf")
                for action in self.Q[stateRecord[N+1]].keys():
                    if self.Q[stateRecord[N+1]][action] > bestReward:
                        bestReward = self.Q[stateRecord[N+1]][action]
                self.Q[stateRecord[N]][actionRecord[N]]\
                    = self.Q[stateRecord[N]][actionRecord[N]] + self.alpha*self.gamma*bestReward

    def playLearningGame(self, epsilon):
        #start game
        self.game.initializeWorld()
        #game history
        stateList = []
        actionList = []
        rewardList = []
        #play the game
        while self.game.liveGame(): #play the game until it is over
            move = ""
            target = ""
            #perceive
            worldState = self.game.getState()
            stateList.append(worldState)
            moveList = self.game.getPossibleActions()
            
            #think
            if worldState not in self.Q.keys():
                #New state
                self.Q.update({worldState: {}})
            numberOfKnownActions = len(self.Q[worldState].keys())
            if numberOfKnownActions == 0: #unknown state
                rnd = np.random.randint(0,len(moveList))
                move = moveList[rnd][0]
                target = moveList[rnd][1]
                self.Q[worldState].update({move+target: 0})
            else: #known state
                #finding the best move (greedy)
                maxScore = float("-inf")
                indMaxScore = -1
                altMoveList = [x[0]+x[1] for x in moveList]
                for i in range(len(moveList)):
                    #initialize if necessary (previously unknown move)
                    if altMoveList[i] not in self.Q[worldState].keys():
                        self.Q[worldState].update({altMoveList[i]: 0})
                    #update best move
                    if maxScore < self.Q[worldState][altMoveList[i]]: #tie is first action in dictionary
                        maxScore = self.Q[worldState][altMoveList[i]]
                        indMaxScore = i
                bestMove = moveList[indMaxScore]

                #Epsilon
                rnd = np.random.rand() #generate a probability
                if rnd < epsilon and len(moveList) is not 1: #Exploration
                    newIndex = indMaxScore
                    while newIndex == indMaxScore: #non-optimal choice selection
                        newIndex = np.random.randint(0,len(moveList))
                    bestMove = moveList[newIndex]
                move = bestMove[0]
                target = bestMove[1]
            #action
            actionList.append(move+target)
            rewardList.append(self.game.updateWorld(move, target))
        return (stateList, actionList, rewardList)

    def playPolicyGame(self, showMaps = False):
        #start game
        self.game.initializeWorld()
        #game history
        stateList = []
        actionList = []
        rewardList = []
        #play the game
        while self.game.liveGame(): #play the game until it is over
            if showMaps:
                print("Game Turn Counter: "+str(self.game.turnCount))
                self.game.printMap()
                print("\n")
            move = ""
            target = ""
            #perceive
            worldState = self.game.getState()
            stateList.append(worldState)
            moveList = self.game.getPossibleActions()
            
            #think
            if worldState not in self.Q.keys():
                #New state
                self.Q.update({worldState: {}})
            numberOfKnownActions = len(self.Q[worldState].keys())
            if numberOfKnownActions == 0: #unknown state
                rnd = np.random.randint(0,len(moveList))
                move = moveList[rnd][0]
                target = moveList[rnd][1]
                self.Q[worldState].update({move+target: 0})
            else: #known state
                #finding the best move (greedy)
                maxScore = float("-inf")
                indMaxScore = -1
                altMoveList = [x[0]+x[1] for x in moveList]
                for i in range(len(moveList)):
                    #initialize if necessary (previously unknown move)
                    if altMoveList[i] not in self.Q[worldState].keys():
                        self.Q[worldState].update({altMoveList[i]: 0})
                    #update best move
                    if maxScore < self.Q[worldState][altMoveList[i]]: #tie is first action in dictionary
                        maxScore = self.Q[worldState][altMoveList[i]]
                        indMaxScore = i
                bestMove = moveList[indMaxScore]
                move = bestMove[0]
                target = bestMove[1]
            #action
            actionList.append(move+target)
            rewardList.append(self.game.updateWorld(move, target))
        return (stateList, actionList, rewardList)

BondGame = escapeGame.escapeRoom(map, timer=80)#Timer was 1000 during training
agent = QLearningAgent(BondGame, alpha=0.05, gamma=0.99, epsilon=0.05)
(stateList, actionList, rewardList) = agent.playPolicyGame()
print(actionList) #random player
#agent.train(10000) Q-the-weapons-guy was trained on 10,000 games
#agent.saveQFunction('Q-the-weapons-guy')
agent.loadQFunction('Q-the-weapons-guy')
agent.playPolicyGame(showMaps=True)