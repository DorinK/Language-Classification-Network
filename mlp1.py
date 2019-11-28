from torch import nn, optim
import torch.nn.functional as F
from trainModel import train_model


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size, hidden_layer_size, output_size):
        super(MultiLayerPerceptron, self).__init__()

        # Linear layers.
        self.linear1 = nn.Linear(input_size, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, output_size)

        # Defining the non linear activation function to be TanH.
        self.activation = nn.Tanh()

    def forward(self, x):

        # For the first linear layer.
        x = self.activation(self.linear1(x))
        x = F.dropout(x, training=self.training)  # Done to prevent over fitting.

        # For the second linear layer.
        x = self.linear2(x)

        return F.log_softmax(x, dim=-1)


if __name__ == '__main__':

    # Creating an instance of MultiLayerPerceptron.
    model = MultiLayerPerceptron(input_size=700, hidden_layer_size=18, output_size=6)

    # Using SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.005)

    # Training the model
    train_model(model, optimizer)
