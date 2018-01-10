# Neural Network: NN class implementation, including forward & back propogation methods 
# Shomik Jain, USC CAIS++

# Neural Networks: directed, weighted graph with layers of neurons/nodes
# each successive layer = increasingly abstract combinations of features from previous layer
# network learns relevant combinations of features to make more accurrate predictions

import numpy as np

# Activation Function: real valued scalar function, squashes output of NN neuron to within desired range 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid (useful during backpropagation) 
def dsigmoid(y):
    return y * (1.0 - y)

# Weight Initialization helper function 
def initialize_weights(d0, d1):
    return np.random.randn(d0, d1)

# Note: NN Neuron -- a = f(z) = f(wp + b) 
# input + weights/bias => activation (change internal state) => output 
class NN(object):
    def __init__(self, layers):
        # a, the output with non-linearity applied.
        self.activations = []
        # z, the weighted output.
        self.z = []
        # W
        self.weights = []
        # The dimensions of our layers.
        self.layers = layers

        # Initialize the dimensions of our layers.
        for layer in layers:
            self.activations.append(np.zeros(layer))
            self.z.append(np.zeros(layer))

        # Initialize the weight values
        for i, layer in enumerate(layers[:-1]):
            self.weights.append(initialize_weights(layers[i + 1], layer))

    # Forward Propagation: propagating inputs through all the neural networ layers => final output 
    def feed_forward(self, inputs):
        # a^0 = x
        self.activation_input = np.array(inputs[:])

        # to keep track of generalized a^i
        a_m = self.activation_input
        self.activations[0] = a_m
        self.z[0] = a_m

        for i, next_weight in enumerate(self.weights):
            # z^(m + 1) = W^(m + 1)a^m
            z_m_next = next_weight.dot(a_m)
            # a^(m + 1) = f(z^(m + 1))
            a_m_next = sigmoid(z_m_next)

            self.activations[i + 1] = a_m_next
            self.z[i + 1] = z_m_next

            a_m = a_m_next

    # Backpropagation: step of propagating error terms backward through the network
    
    # Find error/sensitivity of each neuron (parital derivative of cost w.r.t. neuron's weighted sum)
    # Start at right-most end, propagate error back to neurons in previous layers via chain rule/weights
    # Use sensitivities of each neuron to calculate actual gradients of the parameters (weights)
    # Update parameters (new weights = original - gradient*learning_rate)
    
    def back_propagate(self, targets, learning_rate):
        # delta^L = f'(n^L) * (d J)/(d a)
        # (d J)/(d a) = a
        # In this case (d J)/(d a) = (a^L - t) as we are just using a basic
        # mean squared error.

        # Compute the error to get the final sensitivity term
        error = self.activations[-1] - targets
        delta_L= dsigmoid(self.activations[-1]) * error

        # Start with the final weight transformation
        m = len(self.weights) - 1

        while m >= 0:
            if m != len(self.layers) - 2:
                # delta^m = f'(z^m) * (W^(m+1))^T * detla^(m+1)
                f_prime = dsigmoid(self.activations[m + 1])
                delta_m = f_prime * self.weights[m + 1].T.dot(delta_m_next)
            else:
                delta_m = delta_L

            # W^m (k+1) = W^m(k) - \alpha delta^m * (a^(m-1))^T
            # k is fixed as this is a single iteration.
            self.weights[m] -= learning_rate * delta_m.dot(self.activations[m].T)
            delta_m_next = delta_m
            m -= 1

        error = 0.0

        # Cost Function: Mean Squared Error
        # measures the overall accuracy of the model with current model parameters
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.activations[-1][k]) ** 2

        return error

    # Training: Repeat Forward Propagation + Back Propagation 
    def train(self, patterns, iterations = 3000, learning_rate = 0.001):
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]

                # Compute forward and backwards pass
                self.feed_forward(inputs)
                error = self.back_propagate(targets, learning_rate)

            print('%i: Error %.5f' % (i, error))

data = np.loadtxt('data/pima-indians-diabetes.data.txt', delimiter=',')
X, Y = data[:, :8], data[:, 8]

# Load the data into numpy arrays.
X = np.array(X)
Y = np.array(Y)

# pad with another dimension so we can mutliply by matrices.
X = X.reshape((X.shape[0], X.shape[1], 1))
Y = Y.reshape((Y.shape[0], 1))

# Hidden layer of 20 neurons
nn = NN([X.shape[1], 20, Y.shape[1]])
nn.train(list(zip(X, Y)))


