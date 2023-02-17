import torch
from torch import nn
from torch import optim

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize NeuralNetwork with one output
            * hidden_dims: List of size of dimensions of each layer
                           output layer dimension assumed as 1
            * input_dim: The dimension of the embedding space     
        """
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, 1)
       
    def forward(self, x):
        """
        Computes a forward pass on the network
        returns value in [0, 1]
        """
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = torch.sigmoid(self.layer_3(x))
        return x