from artificial_network_utils import make_matrix
from artificial_network_utils import between


class Layer:

    def __init__(self, id, layer_size, prev_layer_size):

        self.id = id  # this is to help keep track of what layer we are looking at.
        self.n_neurons = layer_size  # how many neurons are in the current layer.
        self.bias_val = 1  # this is the default value for the bias used in order to normalize the threshold

        self.input = [0] * self.n_neurons

        self.output = [0] * (self.n_neurons + self.bias_val)

        self.output[0] = self.bias_val

        self.error = [0] * self.n_neurons

        # make_matrix will generate an N x M matrix
        self.weight = make_matrix(prev_layer_size + self.bias_val, self.n_neurons)

        # This will populate the weight N x M array with randomized values from -0.2 to 0.2
        for i in range(len(self.weight)):
            for j in range(len(self.weight[i])):
                self.weight[i][j] = between(-0.2,0.2)

