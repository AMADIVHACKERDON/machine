import numpy as np
from lawyer import Layer


class Activation(Layer):
    def __init__(self, activation, activation_function_parameters):
        self.activation = activation
        self.activation_function_parameters = activation_function_parameters

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_grad, learning_rate):
        return np.multiply(output_grad, self.activation_function_parameters(self.input))
