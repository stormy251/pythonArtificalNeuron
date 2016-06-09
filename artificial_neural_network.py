from layer import Layer
from artificial_network_utils import sigmoid
from artificial_network_utils import deriv_sigmoid


class ArtificialNeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        self.learn_rate = 0.1

        for i in range(len(layer_sizes)):
            layer_size = layer_sizes[i]
            prev_layer_size = 0 if i == 0 else layer_sizes[i - 1]
            layer = Layer(1, layer_size, prev_layer_size)
            self.layers.append(layer)

    def train(self, inputs, targets, n_epochs):
        """
        Train the network with the labeled inputs for a maximum number of epochs.
        :return:
        """
        for epoch in range(0, n_epochs):

            for i in range(0, len(inputs)):
                self.set_input(inputs[i])
                self.forward_propagate()
                self.update_error_output(targets[i])
                self.backward_propagate()
                self.update_weights()

    def predict(self, input):
        """
        Return the network prediction for this input.
        :return:
        """
        self.set_input(input)
        self.forward_propagate()
        return self.get_output()

    def update_weights(self):
        """
        Update the weight matrix in each layer.
        :return:
        """
        for i in range(1, len(self.layers)):
            for j in range(0, self.layers[i].n_neurons):
                for k in range(0, self.layers[i - 1].n_neurons + self.layers[0].bias_val):
                    out = self.layers[i - 1].output[k]
                    err = self.layers[i].error[j]
                    self.layers[i].weight[k][j] += self.learn_rate * out * err

    def set_input(self, input_vector):
        input_layer = self.layers[0]

        for i in range(0, input_layer.n_neurons):
            input_layer.output[i + input_layer.bias_val] = input_vector[i]

    def forward_propagate(self):
        """
        Propagate the input signal forward through the network.
        :return:
        """

        # exclude the last layer
        for i in range(len(self.layers) - 1):

            src_layer = self.layers[i]
            dst_layer = self.layers[i + 1]

            for j in range(0, dst_layer.n_neurons):

                sum_in = 0

                for k in range(0, src_layer.n_neurons + self.layers[0].bias_val):
                    sum_in += dst_layer.weight[k][j] * src_layer.output[k]

                dst_layer.input[j] = sum_in
                dst_layer.output[j + self.layers[0].bias_val] = sigmoid(sum_in)

    def get_output(self):
        output_layer = self.layers[-1]
        res = [0] * output_layer.n_neurons

        for i in range(0, len(res)):
            res[i] = output_layer.output[i + self.layers[0].bias_val]

        return res

    def update_error_output(self, target_vector):
        output_layer = self.layers[-1]
        for i in range(0, output_layer.n_neurons):
            neuron_output = output_layer.output[i + self.layers[0].bias_val]
            neuron_error = target_vector[i] - neuron_output
            output_layer.error[i] = deriv_sigmoid(output_layer.input[i]) * neuron_error

    def backward_propagate(self):
        """
        Backprop. Propagate the error from the output layer backards to the input layer.
        :return:
        """
        for i in range(len(self.layers) - 1, 0, -1):
            src_layer = self.layers[i]
            dst_layer = self.layers[i - 1]

            for j in range(0, dst_layer.n_neurons):

                error = 0

                for k in range(0, src_layer.n_neurons):
                    error += src_layer.weight[j + self.layers[0].bias_val][k] * src_layer.error[k]

                dst_layer.error[j] = deriv_sigmoid(dst_layer.input[i]) * error
