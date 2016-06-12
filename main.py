from artificial_neural_network import ArtificialNeuralNetwork
from termcolor import colored


logical_artificial_neural_network = ArtificialNeuralNetwork([2, 1])
inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]

# andTargets = [[0.0], [0.0], [0.0], [1.0]]
# orTargets = [[0.0], [1.0], [1.0], [1.0]]
xorTargets = [[0.0], [1.0], [1.0], [0.0]]

targets = xorTargets

# train and predict

print "Prediction WITH any training"
logical_artificial_neural_network.train(inputs, targets, 20000)
for i in range(len(targets)):
    print "Input: ", inputs[i], "Expected output: ", targets[i], logical_artificial_neural_network.predict(inputs[i])