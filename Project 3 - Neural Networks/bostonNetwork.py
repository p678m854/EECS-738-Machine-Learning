import numpy as np
import neuralNetworkClassfile as nn
from sklearn.datasets import load_boston #sklearn has boston data
from random import shuffle
import matplotlib.pyplot as plt

#get data set from sklearn library
bostonData = load_boston()
targets = bostonData['target']
targets = np.asarray(targets)
print(targets)
inputs = bostonData['data']
inputs = np.asarray(inputs)
print(inputs)

#rescaling to [0-1] range for all input/output axis
featureMax = np.amax(inputs,axis = 0)
featureMin = np.amin(inputs,axis = 0)
for j in range(inputs.shape[1]):
    inputs[:,j] -= featureMin[j]
    inputs[:,j] /= (featureMax[j] - featureMin[j])
    
targetMax = np.amax(targets)
targetMin = np.amin(targets)
targets -= targetMin
targets /= (targetMax - targetMin)


#recombining inputs and targets to be pairs
bostonData = list(zip(inputs, targets))

#setting up number of training, validation, and testing sets
num_examples = len(bostonData)
train_frac = int(0.7*num_examples)
valid_frac = int(0.15*num_examples)
test_frac = num_examples - train_frac - valid_frac

#assigning randomly training, validation, and testing sets
shuffle(bostonData)
train_set = bostonData[:train_frac]
valid_set = bostonData[train_frac:(train_frac+valid_frac)]
test_set = bostonData[(train_frac+valid_frac):]

#Training the network
n_inputs = 13
n_hidden = 4
n_output = 1
lr = 0.01
n_epochs = 100

regressionNN = nn.neuralNetwork(n_inputs,n_hidden,n_output,lr)
regressionNN.loadWeights("regressionWeights.p")#Previous weights trained on 20,000 epochs
(e_train, e_valid) = regressionNN.trainRegression(n_epochs,train_set,valid_set)
#regressionNN.saveWeights("regressionWeights.p")
epochs = np.arange(0,n_epochs)

total_error = [np.sqrt(et**2 + ev**2) for et, ev in zip(e_train, e_valid)]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(epochs, e_train,c='r',label="Training error")
ax.plot(epochs, e_valid,c='b',label="Validation error")
ax.plot(epochs, total_error,c='k',label="Total error")
plt.ylabel("RMS error")
plt.xlabel("Epoch number")
plt.legend(loc='upper right')
plt.title("Trainig error rates")

#Generating comparison plots from test dataset
predictedTest = []
for i in range(len(test_set)):
    predictedTest.append(regressionNN.query(test_set[i][0]))

#Denormalizing
realTarget = list(zip(*test_set))[1]
realTarget = np.asarray(realTarget)
realTarget *= (targetMax-targetMin)
realTarget += targetMin

predictedTest = np.asarray(predictedTest)
predictedTest = predictedTest[:,0,0]
predictedTest *= (targetMax-targetMin) #scaling
predictedTest += targetMin  #translating
print(predictedTest.shape)
testNumber = np.arange(0, len(test_set))
print(testNumber.shape)

fig2 = plt.figure()
ax = fig2.add_subplot(1,1,1)
ax.plot(testNumber, predictedTest, c='b', linestyle='None', marker='x', label="Predicted Value")
ax.plot(testNumber, realTarget,c='k', linestyle='None', marker='o', label="Truth value")
plt.xlabel("Test data set index")
plt.ylabel("Median value of home in $1,000")
plt.legend(loc='upper right')
plt.title("Test data set predictions")