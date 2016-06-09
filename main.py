from artificial_neural_network import ArtificialNeuralNetwork


and_artificial_neural_network = ArtificialNeuralNetwork([2, 1])
inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
targets = [[0.0], [0.0], [0.0], [1.0]]

# make predictions with no training
print "Prediction without any training"
for i in range(len(targets)):
    print inputs[i], and_artificial_neural_network.predict(inputs[i])

# train and predict

print "Prediction WITH any training"
and_artificial_neural_network.train(inputs, targets, 20000)
for i in range(len(targets)):
    print(inputs[i], and_artificial_neural_network.predict(inputs[i]))