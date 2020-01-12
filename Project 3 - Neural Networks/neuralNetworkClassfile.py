import numpy as np
import scipy
import pickle

# neural network class definition
class neuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices,
        #      wih and who

        # 1: mean value of the normal distribution - 0.0
        # 2: standard deviation - based on the root of nodes of the upcomming layer ->
        #     pow(self.hnodes, -0.5) --- exponent -0.5 is equal to root of
        # 3: last param builds the grid of the array (self.hnodes, self.inodes)
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate

        # activation function - sigmoid function
        self.activation_function = lambda x: 1/(1+np.exp(-x))

        pass

    # train the neural network
    def weightsAdjustment(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging form final output layer
        final_outputs = self.activation_function(final_inputs)

        # BACKPROPAGATION #

        # error is the (target - actual)
        output_errors = targets - final_outputs

        # hidden layer error is the output_error, split by weights, recombined at hidden nodes
        hidden_errors = np.matmul(self.who.T,(output_errors * final_outputs * (1.0 - final_outputs)))
        #hidden_errors = np.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

        pass

    # query the neural network
    def query(self, inputs_list):
        # convert input list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calcuclate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals  into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def trainRegression(self, epochs, train_set, valid_set):
        num_examples = len(train_set)
        num_validation = len(valid_set)
        rms_train = 0
        rms_valid = 0
        
        e_train_hist = []
        e_valid_hist = []
        for e in range(epochs):
            print('epoch: ',e)
            rms_train = 0
            rms_valid = 0
            # go through all records in the training data set
            for i in range(num_examples):
                self.weightsAdjustment(train_set[i][0].T, train_set[i][1].T)
            #Evaluating the train accuracy at each epoch
            for i in range(num_examples):
                rms_train += ((train_set[i][1] - self.query(train_set[i][0].T)) ** 2)/num_examples
            #Evaluating the validation set accuracy
            for i in range(num_validation):
                rms_valid += ((valid_set[i][1] - self.query(valid_set[i][0].T)) ** 2)/num_validation
            
            rms_train = np.sqrt(rms_train)
            rms_valid = np.sqrt(rms_valid)
            
            e_train_hist.append(rms_train[0,0])
            e_valid_hist.append(rms_valid[0,0])
            print('test rms = ', rms_train[0,0],',\tvalidation rms = ', rms_valid[0,0], 
                ',\ttotal rms = ',np.sqrt((num_examples*(rms_train ** 2) + num_validation*(rms_valid ** 2))
                /(num_examples+num_validation))[0,0])

        return (e_train_hist, e_valid_hist)
    
    def trainClassifier(self, epochs, train_set, valid_set):
        num_examples = len(train_set)
        num_validation = len(valid_set)
        num_corr_train = 0
        num_corr_val = 0

        e_train_hist = []
        e_valid_hist = []
        for e in range(epochs):
            print('epoch: ',e)
            num_corr_train = 0
            num_corr_val = 0
            # go through all records in the training data set
            for i in range(num_examples):
                self.weightsAdjustment(train_set[i][0].T, train_set[i][1].T)
            #Evaluating the train accuracy at each epoch
            for i in range(num_examples):
                if(np.argmax(self.query(train_set[i][0].T)) == np.argmax(train_set[i][1].T)):
                    num_corr_train += 1
            #Evaluating the validation set accuracy
            for i in range(num_validation):
                if(np.argmax(self.query(valid_set[i][0].T)) == np.argmax(valid_set[i][1].T)):
                    num_corr_val += 1


            e_train_hist.append(1-num_corr_train/num_examples)
            e_valid_hist.append(1-num_corr_val/num_validation)
            print('test accuracy = ', 1-e_train_hist[e],',\tvalidation accuracy = ', 1-e_valid_hist[e], 
                ',\ttotal error = ',np.sqrt((e_train_hist[e] ** 2) + (e_train_hist[e] ** 2)))

        return (e_train_hist, e_valid_hist)

    def saveWeights(self, filename = "neuralNetworkWeights.p"):
        dump_file = open(filename, "wb")
        pickle.dump((self.wih, self.who), dump_file)


    def loadWeights(self, filename):
        oldWeights = pickle.load( open(filename, "rb") )
        self.wih = oldWeights[0]
        self.who = oldWeights[1]
