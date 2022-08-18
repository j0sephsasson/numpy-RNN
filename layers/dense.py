import numpy as np

class Dense:
    def __init__(self, neurons, input_shape, activation='softmax'):
        """
        Initializes a simple dense layer

        Args:
            'neurons': int, num of output dimensions
        """
        self.neurons = neurons
        self.name = 'Dense'
        self.activation = activation
        self.weights = np.random.uniform(low=-1, high=1, size=(input_shape, self.neurons))
        self.bias = np.zeros((1, self.neurons))
        
    def softmax(self, inputs):
        """
        Softmax Activation Function used to copute multi-class output probabilities
        """
        exp_scores = np.exp(inputs)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs
        
    def forward(self, inputs):
        """
        Compute forward pass of single (output) layer
        """
        y = np.dot(inputs, self.weights) + self.bias
        
        return self.softmax(y)