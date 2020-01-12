class player:
    def __init__(self, name, health, damage, x, y):
        self.name = name
        self.health = health
        self.damage = damage
        self.position = [x, y]
    def getName(self): return self.name
    def getHealth(self): return self.health
    def getAttackDamage(self): return self.damage
    def getXPosition(self): return self.position[0]
    def getYPosition(self): return self.position[1]
    def receiveDamage(self, damage):
        self.health = max(0,self.health - damage)
    def isDefeated(self):
        if self.health == 0:
            return True
        else:
            return False
    def moveUp(self): self.position[0] = self.position[0] - 1
    def moveDown(self): self.position[0] = self.position[0] + 1
    def moveRight(self): self.position[1] = self.position[1] + 1
    def moveLeft(self): self.position[1] = self.position[1] - 1

class enemy(player):
    #for naming purposes
    enemyCount = 0

    @classmethod
    def updateCount(cls):
        cls.enemyCount += 1
    @classmethod
    def resetCount(cls):
        cls.enemyCount = 0

    def __init__(self, x, y):
        player.__init__(self, "Enemy"+str(self.enemyCount), 10, 2, x, y)
        self.updateCount()

class button:
    buttonCount = 0
    @classmethod
    def updateCount(cls):
        cls.buttonCount +=1
    @classmethod
    def resetCount(cls):
        cls.buttonCount = 0

    def __init__(self, x, y):
        self.name = "Button" + str(self.buttonCount)
        self.updateCount()
        self.position = [x, y]
        self.pushed = False

    def getName(self): return self.name
    def getXPosition(self): return self.position[0]
    def getYPosition(self): return self.position[1]
    def isPushed(self): 
        return self.pushed
    def pushButton(self):
        self.pushed = True
class door:
    doorCount = 0
    @classmethod
    def updateCount(cls):
        cls.doorCount +=1
    @classmethod
    def resetCount(cls):
        cls.doorCount = 0
    def __init__(self, x, y):
        self.id = self.doorCount
        self.updateCount()
        self.position = [x, y]
        self.open = False
    def getID(self): return self.id
    def getXPosition(self): return self.position[0]
    def getYPosition(self): return self.position[1]
    def isOpen(self): return self.open
    def openDoor(self): self.open = True

class hero(player):
    def __init__(self,x,y):
        player.__init__(self, "Bond", 20, 5, x, y)