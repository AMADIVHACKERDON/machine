import math
import random

class Neuron:
    def __init__(self,position_in_layer, is_output_neuron=False):
        self.weights = []
        self.inputs = []
        self.output = None

        self.updated_weights = []
        self.is_output_neurons = is_output_neuron
        self.delta = None
        self.position_in_layer = position_in_layer

    def attach_to_output(self,neurons):
        self.output_neurons = []
        for neuron in neurons:
            self.output_neurons.append(neuron)

    def sigmoid(self,x):
        return 1/ (1+math.exp(-x))
    def init_weights(self,num_input):
        for i in range(num_input):
            self.weights.append(random.uniform(0,1))

    def predict(self,row):
        self.inputs = []

        activation = 0
        for weight, feature in zip(self.weights,row):
            self.inputs.append(feature)
            activation = activation + weight*feature

        self.output = self.sigmoid(activation)
        return self.output
    def update_neuron(self):
        self.weights = []
        for new_weight in self.updated_weights:
            self.weights.append(new_weight)

    def calculate_update(self,learning_rate,target):
        if self.is_output_neurons:
            self.delta = (self.output - target)*self.output*(1-self.output)
        else:
            delta_sum = 0
            cur_weight_index = self.position_in_layer
            for output_neuron in self.output_neurons:
                delta_sum = delta_sum + (output_neuron.delta * output_neuron.weights[cur_weight_index])

            self.delta = delta_sum*self.output*(1-self.output)

        self.updated_weights = []

        for cur_weight,cur_input in zip(self.weights, self.inputs):
            gradient = self.delta*cur_input
            new_weight = cur_weight - learning_rate*gradient
            self.updated_weights.append(new_weight)

class MultilayerPerceptron():
    def __init__(self,num_neuron, learning_rate=0.01, num_iteration = 100):
        self.output_neuron = Neuron(0, is_output_neuron=True)
        self.perceptrons = []
        for i in range(num_neuron):
            neuron = Neuron(i)
            neuron.attach_to_output([self.output_neuron])
            self.perceptrons.append(neuron)

        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.num_neuron = num_neuron

    def fit(self,x,y):
        num_row = len(X)
        num_feature = len(X[0])

        for neuron in self.perceptrons:
            neuron.init_weights(num_feature)
        self.output_neuron.init_weights(len(self.perceptrons))

        for i in range(self.num_iteration):
            r_i = random.randint(0,num_row-1)
            row = X[r_i]
            yhat = self.predict(row)
            target = y[r_i]

            self.output_neuron.calculate_update(self.learning_rate,target)
            for neuron in self.perceptrons:
                neuron.calculate_update(self.learning_rate,target)

                self.output_neuron.update_neuron()
                for neuron in self.perceptrons:
                    neuron.update_neuron()

                if i % 1000 == 0:
                    total_error = 0
                    for r_i in range(num_row):
                        row = X[r_i]
                        yhat = self.predict(row)
                        error = (y[r_i] - yhat)
                        total_error = total_error + error**2
                    mean_error = total_error/num_row
                    print(f'Iteration {i} with error = {mean_error}')

    def predict(self,row):
        row.append(1)
        activations = [perceptron.predict(row) for perceptron in self.perceptrons]
        activations.append(1)
        activation = self.output_neuron.predict(activations)

        if activation >= 0.5:
            return 1.0
        return 0.0

X = [[0,0],[0,1],[1,0],[1,1]]
y = [0,1,1,0]
num_neuron = 10
num_iteration = 150000
clf = MultilayerPerceptron(num_neuron, num_iteration=num_iteration)
clf.fit(X,y)