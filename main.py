# Import section
from termcolor import colored

from storm_ANN.artificial_neural_network import ArtificialNeuralNetwork

# Declaring Neural Network
logical_artificial_neural_network = ArtificialNeuralNetwork([2, 1])

# Defining inputs to the neural network
inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

# Dictionary of different types of target logical functions.
targetObj = {
    "andTargets": [[0.0], [0.0], [0.0], [1.0]],
    "orTargets": [[0.0], [1.0], [1.0], [1.0]],
    "xorTargets": [[0.0], [1.0], [1.0], [0.0]]
}

# Select the target output that you would like
targets = targetObj["andTargets"]

# Use the ANN to train with the given inputs and the expected outputs. train (trainingIterations) times.
training_iterations = 20000
num_of_targets = len(targets)

logical_artificial_neural_network.train(inputs, targets, training_iterations)
for i in range(num_of_targets):
    print \
        colored("Input --", "blue", attrs=['bold']), inputs[i], \
        colored("Expected output --", "red", attrs=['bold']), targets[i], \
        colored("ANN Output --", "green", attrs=['bold']), logical_artificial_neural_network.predict(inputs[i])
