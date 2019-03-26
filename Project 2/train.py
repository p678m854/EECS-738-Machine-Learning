import string
import numpy as np
import csv
import re

#Get Data
trainingFile = './Henry IV.csv'

#form dictionary to be able to compare two given states (markov)
def form_dict(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = {}
    if value not in dictionary[key].keys():
        dictionary[key][value] = 0    
    dictionary[key][value] += 1

def formMultipleFreqDictionary(dictionary, dictKey, key, value):
    if dictKey not in dictionary:
        dictionary[dictKey] = {}
    if dictKey not in actorList: #Im assuming your only using this with actors
        actorList.append(dictKey)
    if key not in dictionary[dictKey].keys():
        dictionary[dictKey][key] = {}
    if value not in dictionary[dictKey][key]:
        dictionary[dictKey][key][value] = 0
    dictionary[dictKey][key][value] += 1

#We have to have a place for the first and second words because they will not have a key value pair
#aLso need to have a place when we transition to next line

#Essentially these dictionaries hold the conditional probabilities for words by an actor
first_word = {}
second_word = {}
transition = {}

#You can consider the actors as the true states and the words the emmissions
actorMM = {} #state transition for the words
actorList = []
trainingFile = './Project 2/Shakespeare_data.csv'

def actSceneLine(strASL):
    strASL = strASL.split('.')
    scene = int(strASL[1])
    return scene

def returnLine(strLine):
    return re.findall(r"[\w]+|[.,!?:]",strLine)

def actorTrain(trainingFile):
    with open(trainingFile) as csvfile: #open up Shakespeare
        reader = csv.DictReader(csvfile)
        prvScene = 0
        previousActor = 'START SCENE'

        for row in reader:
            if row['ActSceneLine'] != '': # Lines only delivered

                #Update check for new play scene
                scene = actSceneLine(row['ActSceneLine'])
                if scene != prvScene:
                    if previousActor != 'START SCENE': # Exception only at beginning of file
                        form_dict(actorMM, previousActor, 'END SCENE')
                        form_dict(actorMM, 'END SCENE', 'START SCENE')
                        previousActor = 'START SCENE' # No previous actor at start of scene 
                prvScene = scene

                # Forming model for actors
                currentActor = row['Player']
                currentLine = returnLine(row['PlayerLine'])

                form_dict(actorMM,previousActor,currentActor)
                previousActor = currentActor

                # Forming model for player lines
                num_words = len(currentLine)
                for i in range(num_words):
                    word = currentLine[i]
                    if i != 0:
                        previous_word = currentLine[i - 1]
                    if i == 0:
                        formMultipleFreqDictionary(first_word, currentActor, 'NEXT LINE', word)
                    #if index is 1 then it is the second word,  need to add to dictionary
                    elif i == 1:
                        formMultipleFreqDictionary(second_word, currentActor, previous_word, word)
                    else:
                        two_words_ago = currentLine[i - 2]
                        formMultipleFreqDictionary(transition, currentActor, (two_words_ago, previous_word), word)

                    #if index is last word then we need to mark it with 'Next Line' in the dictionary
                    if i == num_words - 1:
                        formMultipleFreqDictionary(transition, currentActor, word, 'NEXT LINE') 
        
        #Normalizing the actor model distribution
        for currentActor, nextActor in actorMM.items():
            total_events = 0
            for na in nextActor:
                total_events += actorMM[currentActor][na]
            for na in nextActor:
                actorMM[currentActor][na] /= total_events
        
        # We have to Normalize the word distributions (maximum likelihood estimation)
        for dictionary in [first_word, second_word, transition]:
            #Going through every action in the dictionary
            for actor in actorList:
                #Sometimes an actor only has 1 line ever
                if actor in dictionary.keys():
                    # For every word/set of words
                    for key, possibleValue in dictionary[actor].items():
                        #Summation loop to find total events
                        dictionary_total = 0
                        for value in possibleValue:
                            dictionary_total += dictionary[actor][key][value]
                        #Loop to get realative frequency
                        for value in possibleValue:
                            dictionary[actor][key][value] /= dictionary_total

        #for previousActor, currentActor in actorMM.items():
         #   actorMM

    print("training finished")                                  

#We Have to get a random word to generate our text
def random_word(dictionary):
    p0 = np.random.random()
    cumulative = 0
    for key, value in dictionary.items():
        cumulative += value
        if p0 < cumulative:
            return key
    assert(False)

def generate_text(dictKey):
    spaceDeliminator = " "
    sentence = []
    i = 0

    MLword = ''
    prevWord = ''
    prevPrevWord = ''
    while True:
        #reset probability
        MLprob = 0
        
        if i == 0:
            for word in first_word[dictKey]['NEXT LINE'].keys():
                if first_word[dictKey]['NEXT LINE'][word] > MLprob:
                    MLprob = first_word[dictKey]['NEXT LINE'][word]
                    MLword = word
        elif i == 1:
            prevWord = MLword
            MLprob = 0
            for word in second_word[dictKey][prevWord].keys():
                if second_word[dictKey][prevWord][word] > MLprob:
                    MLprob = second_word[dictKey][prevWord][word]
                    MLword = word
        else:
            prevPrevWord = prevWord
            prevWord = MLword
            if (prevPrevWord, prevWord) in transition[dictKey].keys():
                for word in transition[dictKey][(prevPrevWord, prevWord)].keys():
                    if transition[dictKey][(prevPrevWord, prevWord)][word] > MLprob:
                        MLprob = transition[dictKey][(prevPrevWord, prevWord)][word]
                        MLword = word
            elif prevWord in second_word[dictKey].keys():
                for word in transition[dictKey][prevWord].keys():
                    if transition[dictKey][(prevPrevWord, prevWord)][word] > MLprob:
                        MLprob = transition[dictKey][(prevPrevWord, prevWord)][word]
                        MLword = word
            else:
                MLword = 'NEXT LINE'

            if MLword in transition[dictKey].keys():
                if 'NEXT LINE' in transition[dictKey][MLword].keys():
                    if MLprob < transition[dictKey][word]['NEXT LINE']:
                        MLword = 'NEXTLINE'

        if MLword == 'NEXT LINE':
            break
        else:
            if i != 0:
                if MLword not in [',', '?', '!', '.', ':']:
                    sentence.append(spaceDeliminator)
            sentence.append(MLword)
        i += 1

    print(''.join(sentence))

#Pretty much following wikipedia's pseudo code
def actorViterbi(testFile):
    #open text file
    lines = open(testFile)
    
    # Initializing state path matrices
    num_lines = sum(1 for line in lines)
    num_actors = len(actorList)
    T1 = np.zeros((num_actors,num_lines+1))
    T2 = np.zeros((num_actors,num_lines+1), dtype=int)
    for i in range(num_actors):
        T1[i,0] = 1/num_actors #assuming uniform

    lines = open(testFile)
    word = ''
    prevWord = ''
    prevPrevWord = ''
    i = 1 #Manually indexing of time
    #Iterate through the lines
    for line in lines:
        currentLine = returnLine(line)

        #Fill out T1 and T2 matrices
        for j in range(num_actors):
            MLindex = 0 #index
            MLvalue = 0 #probability    
            possibleActor = actorList[j]
            cumProbabilityProduct = 1
            num_words = len(currentLine)

            #The emmissions matrix component (B_{j,y_{i})
            for k in range(num_words):
                if k == 0:
                    word = currentLine[k]
                    cumProbabilityProduct *= first_word[possibleActor]['NEXT LINE'].get(word,0)
                elif k == 1:
                    word = currentLine[k]
                    prevWord = currentLine[k-1]
                    if possibleActor in second_word.keys():
                        if prevWord in second_word[possibleActor].keys():
                            cumProbabilityProduct *= second_word[possibleActor][prevWord].get(word,0)
                        else:
                            cumProbabilityProduct *= 0
                    else:
                        cumProbabilityProduct *= 0 #Maximum likelihood method estimate
                else:
                    word = currentLine[k]
                    prevWord = currentLine[k-1]
                    prevPrevWord = currentLine[k-2]

                    if possibleActor in transition.keys():
                        if (prevPrevWord, prevWord) in transition[possibleActor].keys():
                            cumProbabilityProduct *= transition[possibleActor][(prevPrevWord, prevWord)].get(word,0)
                        else:
                            cumProbabilityProduct *= 0
                    else:
                        cumProbabilityProduct = 0
                if k == (num_words-1):
                    word = currentLine[k]
                    if word in transition[possibleActor].keys(): 
                        cumProbabilityProduct *= transition[possibleActor][word].get('NEXT LINE',0)
                    else:
                        cumProbabilityProduct *= 0
            
            #transition matrix component (T[k,i-1]*A_{k,j})
            for k in range(num_actors):
                prevActor = actorList[k]
                transLH = T1[k,i-1]*actorMM[prevActor].get(possibleActor,0)

                if transLH > MLvalue:
                    MLvalue = transLH
                    MLindex = k
            
            #Storing T1 and T2 ResultsE
            T1[j,i] = MLvalue*cumProbabilityProduct
            T2[j,i] = MLindex
        i +=1 #manually iterate

    #Time to do most likely path
    z = np.zeros((num_lines+1,1))
    x = [None] * (num_lines+1)
    for i in range(T1.shape[1]):
        if i == 0:
            z[num_lines-i,0] = np.argmax(T1[:,num_lines], axis = 0)
        else:
            z[num_lines-i,0] = T2[int(z[num_lines-i+1,0]),num_lines-i]
    for i in range(z.shape[0]):
        x[i] = actorList[int(z[i,0])]

    #return most likely path
    return x

#Used when evaluating Viterbi algorithm
def listDiff(x,y):
    count = 0
    for i in range(len(x)):
        if x[i] != y[i]:
            count += 1
    
    return count


actorTrain(trainingFile)

#Demos of Viterbi algorithm

print("Demoing Viterbi Algorithm. Hidden states are the actors speaking so we are trying to just the line to which actor")

x1 = actorViterbi('./firstLinesHenryIV.txt')
print("\nFrom the first lines of Henry IV when only Henry IV is speaking: ")
print("Actor speaking: ", x1)

x2 = actorViterbi('./henryIV13.txt')
correctX2 = ['HOTSPUR', 'HOTSPUR', 'HOTSPUR', 'HOTSPUR',\
    'NORTHUMBERLAND', 'NORTHUMBERLAND', 'NORTHUMBERLAND',\
    'HOTSPUR', 'HOTSPUR', 'HOTSPUR', 'HOTSPUR', 'HOTSPUR', 'HOTSPUR', 'HOTSPUR', 'HOTSPUR',\
    'NORTHUMBERLAND',\
    'EARL OF WORCESTER',\
    'HOTSPUR', 'HOTSPUR', 'HOTSPUR', 'HOTSPUR', 'HOTSPUR',\
    'EARL OF WORCESTER', 'EARL OF WORCESTER',\
    'NORTHUMBERLAND', 'NORTHUMBERLAND', 'NORTHUMBERLAND', 'NORTHUMBERLAND', 'NORTHUMBERLAND', 'NORTHUMBERLAND',\
    'EARL OF WORCESTER', 'EARL OF WORCESTER']
print("\nFrom Act 1, Scene 3, Lines 127-155 of Henry IV: ")
print("Correct state path: ", correctX2)
print("\nViterbi state path: ", x2)
print("Ratio correct: ", 1 - (listDiff(x2, correctX2)/len(x2)))

print("\nPredicting the next line (1.3.156): ")
generate_text(x2[-1])
