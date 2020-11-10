import numpy as np
import scipy.special
import matplotlib.pyplot

class neuralnetwork:
    # initialise the neural network
    def __init__(self, inputnodes, hiddenodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddenodes
        self.onodes = outputnodes

        # learning rate
        self.lr = learningrate

        # link weight matrices, wih and who
        # weights inside the  arrays are w_i_j,where link is from node i to node j in the next layer
        # self.wih = (np.random.rand(self.hnodes,self.inodes)-0.5)
        # self.who = (np.random.rand(self.onodes,self.hnodes)-0.5)
        self.wih = np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))

        # activation function is the sigmoid function
        self.activation_function = lambda x:scipy.special.expit(x)
        pass
    # train the neural network
    def train(self, inputs_list, targets_list):

        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layers
        hidden_inputs = np.dot(self.wih, inputs)

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # error is the (target - actual)
        output_errors = targets-final_outputs

        # hidden layer error is the output_errors,split by weights,recombined at hidden nodes
        # 误差的反向传播
        hidden_errors = np.dot(self.who.T,output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr*np.dot((output_errors*final_outputs*(1.0-final_outputs)),np.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass
    # query the neural network
    def query(self,inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layers
        hidden_inputs = np.dot(self.wih, inputs)

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

        pass

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate is 0.3
learning_rate = 0.1

# create instance of neural network
n = neuralnetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the minst training data csv file into a list
training_data_file = open("mnist_train.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network

# go through all records in the training data set
epochs = 5
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:])/255.0*0.99)+0.01
        targets = np.zeros(output_nodes)+0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
    pass
pass
# load the minst test data csv file into a list
test_data_file = open("mnist_test.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

# all_values = test_data_list[0].split(',')
# print(all_values[0])

# image_array = np.asfarray(all_values[1:]).reshape((28,28))
# matplotlib.pyplot.imshow(image_array,cmap='Greys',interpolation='None')
# matplotlib.pyplot.show()

# print(n.query((np.asfarray(all_values[1:])/255.0*0.99)+0.01))

# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the data set
for record in test_data_list:
    all_values = record.split(',')
    correct_lable = int(all_values[0])
    print(correct_lable, "correct label")
    inputs = (np.asfarray(all_values[1:])/255.0*0.99)+0.01
    outputs = n.query(inputs)
    label = np.argmax(outputs)
    print(label, "network's answer")
    if(label == correct_lable):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass
print(scorecard)
scorecard_array = np.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)