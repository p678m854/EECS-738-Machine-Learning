import numpy as np # linear algebra
import neuralNetworkClassfile as nn
import gzip
import pickle
import matplotlib.pyplot as plt

# visualizations
import matplotlib

#One hot encoding of the digits
def oheDigits(j):
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

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

train_set, valid_set, test_set = loadData()

# number of nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# learning rate with 0.1
learning_rate = 0.1

# create an instance of neuralnetwork
network = nn.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# train the neural network
# Load Old Weights
network.loadWeights("neuralNetworkWeights.p")
# epochs -> the number of times the training data set is used for training
epochs = 2
(e_train_hist, e_valid_hist) = network.trainClassifier(epochs, train_set, valid_set)
#network.saveWeights("neuralNetworkWeights.p")
# Visualize loss history
num_e = np.arange(0,epochs)
plt.plot(num_e+1, e_train_hist, 'b')
plt.plot(num_e+1, e_valid_hist, 'g')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss (Percentage)')
plt.show();

#Test verification

correlation_matrix = np.zeros((10,10), dtype = int)
for n in range(len(test_set)):
    i = test_set[n][1]
    j = np.argmax(network.query(test_set[n][0].T))
    correlation_matrix[i,j] = correlation_matrix[i,j] +1
    
print(correlation_matrix)