from storm_ANN.artificial_network_utils import between
from storm_ANN.artificial_network_utils import make_matrix


class Layer:
    def __init__(self, layer_id, layer_size, prev_layer_size):

        self.layer_id = layer_id
        self.n_neurons = layer_size

        # this is the default value for the bias used in order to normalize the threshold
        self.bias_val = 1

        self.input = [0] * self.n_neurons

        self.output = [0] * (self.n_neurons + self.bias_val)

        self.output[0] = self.bias_val

        self.error = [0] * self.n_neurons

        # make_matrix will generate an N x M matrix
        self.weight = make_matrix(prev_layer_size + self.bias_val, self.n_neurons)

        # This will populate the weight N x M array with randomized values from -0.2 to 0.2
        for i in range(len(self.weight)):
            for j in range(len(self.weight[i])):
                self.weight[i][j] = between(-1.0, 1.0)
