import worldObjects
import copy

class escapeRoom:
    def __init__(self, map, timer = 100):
        #initialize world
        self.originalMap = copy.deepcopy(map)
        self.currentMap = copy.deepcopy(map)
        #Create enemies, players, and objects
        self.enemyList = []
        self.playerList = []
        self.buttons = []
        self.doors = []
        self.gameOver = True
        self.turnCount = 0
        self.timer = timer
        self.initializeWorld()

    def printMap(self):
        for line in self.currentMap:
            output = ''
            for tile in line:
                output += '\t'
                if tile != "o":
                    output += tile
            print(output)

    def liveGame(self): return not self.gameOver

    def initializeWorld(self):
        #reinitialize map and game status
        self.currentMap = copy.deepcopy(self.originalMap)
        self.gameOver = False
        self.turnCount = 0    
        #Create enemies, players, and objects
        worldObjects.enemy.resetCount()
        worldObjects.button.resetCount()
        worldObjects.door.resetCount()
        self.enemyList = []
        self.heroList = []
        self.buttons = []
        self.doors = []
        #Loop through map
        for i in range(len(self.originalMap)):
            for j in range(len(self.originalMap[i])):
                tile = self.currentMap[i][j]
                if tile == "enemy":
                    self.enemyList.append(worldObjects.enemy(i,j))
                elif tile == "Bond":
                    self.heroList.append(worldObjects.hero(i,j))
                elif tile ==  "button":
                    self.buttons.append(worldObjects.button(i,j))
                elif tile == "door":
                    self.doors.append(worldObjects.door(i,j))

    #L1 distance metric
    def inPushRange(self, hero, button):
        distance = abs(hero.getXPosition() - button.getXPosition()) \
            + abs(hero.getYPosition() - button.getYPosition())
        if distance <= 1: return True
        else: return False

    #L1 distance metric
    def enemyInCombatRange(self, hero, enemy):
        distance = abs(hero.getXPosition() - enemy.getXPosition()) \
            + abs(hero.getYPosition() - enemy.getYPosition())
        if distance <= 2: return True
        else: return False

    #Linear line of sight along cardinal directions
    def inLOS(self, p1, p2):
        if p1.getXPosition() == p2.getXPosition():
            blocked = False
            startY = 0
            stopY = 0
            if p1.getYPosition() > p2.getYPosition():
                startY = p2.getYPosition() +1
                stopY = p1.getYPosition()
            else:
                startY = p1.getYPosition() +1
                stopY = p2.getYPosition()
            for j in range(startY,stopY):
                if self.currentMap[p1.getXPosition()][j] != "o":
                    blocked = True
                    break
            return not blocked
        elif p1.getYPosition() == p2.getYPosition():
            blocked = False
            startX = 0
            stopX = 0
            if p1.getXPosition() > p2.getXPosition():
                startX = p2.getXPosition() +1
                stopX = p1.getXPosition()
            else:
                startX = p1.getXPosition() +1
                stopX = p2.getXPosition()
            for i in range(startX,stopX):
                if self.currentMap[i][p1.getYPosition()] != "o":
                    blocked = True
                    break
            return not blocked
        else:
            return False

    def updateWorld(self, action, target = None):
        self.turnCount += 1
        reward = -1
        """
        #Rewarding state of world towards finish
        numButtonPressed = sum([1 if button.isPushed() else 0 for button in self.buttons])
        numDoorsOpened = sum([1 if door.isOpen() else 0 for door in self.doors])
        reward -= 10*(5-numButtonPressed)
        """
        #Bond action
        if action == "attack":
            for enemy in self.enemyList:
                if enemy.getName() == target:
                    enemy.receiveDamage(self.heroList[0].getAttackDamage())
                    reward += 10
                    if enemy.isDefeated() is True:
                        reward += 20
                        self.currentMap[enemy.getXPosition()][enemy.getYPosition()] = "o"
        elif action == "push":
            for button in self.buttons:
                if button.getName() == target:
                    button.pushButton()
                    reward += 10
                    self.currentMap[button.getXPosition()][button.getYPosition()] = "o"
            #Update doors in the world
            unPushedButtons = sum([0 if button.isPushed() else 1 for button in self.buttons])
            if unPushedButtons < 2 and not self.doors[1].isOpen():
                self.doors[1].openDoor()
                self.currentMap[self.doors[1].getXPosition()][self.doors[1].getYPosition()] = "o"
                reward += 10
            elif unPushedButtons < 4 and not self.doors[0].isOpen():
                self.doors[0].openDoor()
                self.currentMap[self.doors[0].getXPosition()][self.doors[0].getYPosition()] = "o"
                reward += 10

        elif action in ["moveUp", "moveLeft", "moveRight", "moveDown"]:
            #update hero's old spot 
            xHero = self.heroList[0].getXPosition()
            yHero = self.heroList[0].getYPosition()
            self.currentMap[xHero][yHero] = "o"

            #move hero
            if action == "moveUp":
                self.heroList[0].moveUp()
            elif action == "moveDown":
                self.heroList[0].moveDown()
            elif action == "moveRight":
                self.heroList[0].moveRight()
            elif action == "moveLeft":
                self.heroList[0].moveLeft()
            
            #update new hero's spot
            xHero = self.heroList[0].getXPosition()
            yHero = self.heroList[0].getYPosition()
            self.currentMap[xHero][yHero] = self.heroList[0].getName()

        #game ends on player exiting the stage
        if action == "exit":
            reward += 1000
            self.gameOver = True
        else: #if no exit, world reacts
            #World action
            #Enemy's attack
            for enemy in self.enemyList:
                for hero in self.heroList:
                    if self.inLOS(enemy, hero)\
                        and self.enemyInCombatRange(enemy,hero)\
                        and not enemy.isDefeated():
                        hero.receiveDamage(enemy.getAttackDamage())
                        reward -= 2
                        if hero.isDefeated():
                            reward -= 20
                            self.gameOver = True
            if self.turnCount == self.timer:
                self.gameOver = True
                reward -= 1000

        return reward

    def getPossibleActions(self):
        possibleActions = []
        for hero in self.heroList:
            #Moving
            i = hero.getXPosition()
            j = hero.getYPosition()
            if self.currentMap[i-1][j] == "o" and i > 0:
                possibleActions.append(["moveUp", hero.getName()])
            if self.currentMap[i+1][j] == "o" and i < len(self.currentMap):
                possibleActions.append(["moveDown", hero.getName()])
            if self.currentMap[i][j-1] == "o" and j > 0:
                possibleActions.append(["moveLeft", hero.getName()])
            if self.currentMap[i][j+1] == "o" and j < len(self.currentMap[0]):
                possibleActions.append(["moveRight", hero.getName()])
            #Button pressing
            for button in self.buttons:
                if self.inPushRange(hero, button) and not button.isPushed():
                    possibleActions.append(["push", button.getName()])
            #Attacking options
            for enemy in self.enemyList:
                if self.inLOS(hero, enemy) and self.enemyInCombatRange(hero, enemy) and not enemy.isDefeated():
                    possibleActions.append(["attack", enemy.getName()])
            #Game ending options
            if sum([1 if button.isPushed() else 0 for button in self.buttons]) == len(self.buttons)\
                and hero.getXPosition() == (len(self.currentMap)-2) and hero.getYPosition() == round(len(self.currentMap[0])/2):
                possibleActions.append(["exit", hero.getName()]) 
        return possibleActions

    def getState(self):
        worldState = []
        for hero in self.heroList:
            worldState.append(str(hero.position))
        for button in self.buttons:
            worldState.append("Pressed" if button.isPushed() else "Unpressed")
        for door in self.doors:        
            worldState.append("Opened" if door.isOpen() else "Closed")
        for hero in self.heroList:
            for enemy in self.enemyList:
                if self.inLOS(enemy, hero) and not enemy.isDefeated() and enemy.getName() not in worldState:
                    worldState.append(enemy.getName())
        return ", ".join(worldState) #One string specifically to act as a key in the Q learning table
