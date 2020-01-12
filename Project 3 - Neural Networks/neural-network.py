import gzip #for the nmist dataset
import pickle #already been pickled for us
import numpy as np
from math import exp

#In theory we should add more images to the dataset so that we have an unknown category

# Loading data in
def loadData():
    # Load dataset    
    f = gzip.open('mnist.pkl.gz', 'rb') #deeplearning.net/tutorial/gettingstarted.html
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    
    #Inputs
    train_set_inputs = [np.reshape(x, (784, 1)) for x in train_set[0]]
    valid_set_inputs = [np.reshape(x, (784, 1)) for x in valid_set[0]]
    test_set_inputs = [np.reshape(x, (784, 1)) for x in test_set[0]]
    
    #Labels (One-hot-encodings for both)
    train_set_labels = [oheDigits(y) for y in train_set[1]]
    valid_set_labels = [oheDigits(y) for y in valid_set[1]]
    test_set_labels = [y for y in test_set[1]]

    #repackaging inputs and labels together
    train_set = list(zip(train_set_inputs, train_set_labels))
    valid_set = list(zip(valid_set_inputs, valid_set_labels))
    test_set = list(zip(test_set_inputs, test_set_labels))

    return (train_set, valid_set, test_set)

#One hot encoding of the digits
def oheDigits(j):
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

def ReLU(x):
    return max(x,0)

def softMax(xVector):
    outputVector = np.zeros((xVector.shape[0],1))
    totalWeights = 0
    for i in range(xVector.shape[0]):
        expValue = exp(xVector[i,0])
        outputVector[i,0] = expValue
        totalWeights += expValue
    for i in range(xVector.shape[0]):
        outputVector[i,0] = outputVector[i,0]/expValue

    return outputVector

def kroneckerDelta(i,j):
    if i == j:
        return 1
    else:
        return 0

def gradientCrossEntropy(inputVector, targetVector):
    # \partial L/ \partial o_i = p_i - y_i where o_i is the ith network output (before softmax)
    return inputVector-targetVector


class neuralNetwork:
    def __init__(self, inputsize, outputsize, hiddenLayers = 0, sizeOfHiddenLayers = 0, activation = "ReLU", nettype = "classification", learningRate = 0):
        #setting up size of hidden layers
        self.nLayers = hiddenLayers + 2 #Input+Output+HiddenLayers
        self.weights = [] #weights length will end up being 1 less than number of layers
        self.bias = []
        self.activationFunction = activation
        self.type = nettype
        
        #Preallocating weights and bias
        for i in range(self.nLayers - 1):
            #weights
            if i == 0: #first connection
                if self.nLayers == 2: #straight input->output
                    self.weights.append(np.zeros((outputsize, inputsize)))
                else: # input -> hidden layer
                    self.weights.append(np.zeros((sizeOfHiddenLayers, inputsize)))
            elif i == (self.nLayers -2): # last connection 
                self.weights.append(np.zeros((outputsize, sizeOfHiddenLayers))) # hidden -> output
            else:
                self.weights.append(np.zeros((sizeOfHiddenLayers, sizeOfHiddenLayers))) # hidden -> hidden
            
            #biases
            if i == (self.nLayers-2): #output
                self.bias.append(np.zeros((outputsize,1)))
            else: #hidden layer
                self.bias.append(np.zeros((sizeOfHiddenLayers,1)))

    def forwardPropogate(self, inputVector):
        for n in range(self.nLayers-1):
            #define intermediate results
            tempResults = np.zeros((self.weights[n].shape[0],1)) #intermidate vector
            for i in range(tempResults.shape[0]):
                tempResults[i,0] = np.matmul(self.weights[n][i,:],inputVector) + self.bias[n][i,0]
                tempResults[i,0] = eval(self.activationFunction + "(" + str(tempResults[i,0]) + ")")

            if n == (self.nLayers - 2):
                if self.type == "classification":
                    tempResults = softMax(tempResults)
                return tempResults
            else:
                inputVector = tempResults

    def backpropagation(self, X, y, learning_rate):
        """
        Performs the backward propagation algorithm and updates the layers weights.
        :param X: The input values.
        :param y: The target values.
        :param float learning_rate: The learning rate (between 0 and 1).
        """

        # Feed forward for the output
        output = self.forwardPropogate(X)

        # Loop over the layers backward
        for i in reversed(range(len(self.nLayers))):
            layer = self.nLayers[i]

            # If this is the output layer
            if layer == self.nLayers[-1]:
                layer.error = y - output
                # The output = layer.last_activation in this case
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:
                next_layer = self.nLayers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)

        # Update the weights
        for i in range(len(self.nLayers)):
            layer = self.nLayers[i]
            # The input is either the previous layers output or X itself (for the first hidden layer)
            input_to_use = np.atleast_2d(X if i == 0 else self.nLayers[i - 1].ReLU)
            layer.weights += layer.delta * input_to_use.T * learning_rate

train_set, valid_set, test_set = loadData()

myNN = neuralNetwork(784, 10, hiddenLayers=2, sizeOfHiddenLayers=10)
print(myNN.forwardPropogate(train_set[0][0]))
