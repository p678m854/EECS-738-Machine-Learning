import string
import numpy as np





#Get Data
training_data = open('alllines.txt')

#Remove Commas
def remove_commas(sentence):
    return sentence.translate(str.maketrans('', '', string.punctuation))



#form dictionary to be able to compare two given states (markov)
def form_dict(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = []
    dictionary[key].append(value)

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

first_word = {}
second_word = {}
transition = {}

#Train our model
def train():
    # we need to grab our our words to process, remove commas and make lowercase
    for line in training_data:

        words = remove_commas(line.rstrip().lower()).strip()
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

    # We have to Normalize the distributions
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
    for i in range(num_lines):
        sentence = []
        # Fist word
        word0 = random_word(first_word)
        sentence.append(word0)
        # Second word
        word1 = random_word(second_word[word0])
        sentence.append(word1)
        #Remaining words until NEW LINE
        while True:
            word2 = random_word(transition[(word0, word1)])
            if word2 == "NEXT LINE":
                break
            sentence.append(word2)
            word0 = word1
            word1 = word2
        print(''.join(sentence))

train()
generate_text()
