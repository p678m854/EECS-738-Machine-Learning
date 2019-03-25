import string
import numpy as np
import csv
import re

#Get Data
training_data = open('./Project 2/alllines.txt')


#Remove Commas
def remove_commas(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))

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
        actorList.append(dictKey)
    if key not in dictionary[dictKey].keys():
        dictionary[dictKey][key] = {}
    if value not in dictionary[dictKey][key]:
        dictionary[dictKey][key][value] = 0
    dictionary[dictKey][key][value] += 1


#We need to keep track of probabilities from each two pairs of words
def probabilities(list):
    probability_list= {}
    length_of_list = len(list)
    for i in list:
        probability_list[i] = probability_list.get(i, 0) + 1
    for key, value in probability_list.items():
        probability_list[key] = value / length_of_list
    return probability_list

#We have to have a place for the first and second words because they will not have a key value pair
#aLso need to have a place when we transition to next line

#Essentially these dictionaries hold the marginal probabilities for words by an actor
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


#Train our model
def train():
    # we need to grab our our words to process, remove commas and make lowercase
    for line in training_data:

        words = remove_commas(line.rstrip().lower()).strip()
        words = words.split()
       #Create indexes for words
        num_words = len(words)
        for i in range(num_words):
            word = words[i]
            #if index is zero, that means it is the first word
            if i == 0:
                first_word[word] = first_word.get(word, 0) + 1
            else:
                #create index for previous word
                previous_word = words[i - 1]
                #if index is last word then we need to mark it with 'Next Line' in the dictionary
                if i == num_words - 1:
                    form_dict(transition, (previous_word, word), 'NEXT LINE')
                #if index is 1 then it is the second word,  need to add to dictionary
                if i == 1:
                    form_dict(second_word, previous_word, word)
                else:
                    two_words_ago = words[i - 2]
                    form_dict(transition, (two_words_ago, previous_word), word)

    # We have to normalize the word distributions
    total_actors = sum(actorMM.values())
    for key, value in actorMM.items():
        actorMM[key] = value/total_actors

    first_word_total = sum(first_word.values())
    for key, value in first_word.items():
        first_word[key] = value / first_word_total

    for previous_word, next_word in second_word.items():
        second_word[previous_word] = probabilities(next_word)

    for two_words , next_word in transition.items():
        transition[two_words]= probabilities(next_word)

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


num_lines = 10

def generate_text():
    spaceDeliminator = " "
    for i in range(num_lines):
        sentence = []
        # Fist word
        word0 = random_word(first_word)
        sentence.append(word0)
        # Second word
        word1 = random_word(second_word[word0])
        sentence.append(spaceDeliminator)
        sentence.append(word1)
        #Remaining words until NEW LINE
        while True:
            word2 = random_word(transition[(word0, word1)])
            if word2 == "NEXT LINE":
                break
            sentence.append(spaceDeliminator)
            sentence.append(word2)
            word0 = word1
            word1 = word2
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
            
            #Storing T1 and T2 Results
            T1[j,i-1] = MLvalue*cumProbabilityProduct
            T2[j,i-1] = MLindex
        i +=1 #manually iterate

    #Time to do most likely path
    print(T1)
    print(T2)
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

actorTrain(trainingFile)

#Demos of Viterbi algorithm
print(actorViterbi('./Project 2/firstLinesHenryIV.txt'))
print(actorViterbi('./Project 2/kingLear22.txt'))
#train()
#generate_text()