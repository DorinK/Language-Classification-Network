from torch import nn, optim
import torch.nn.functional as F
from trainModel import train_model


class LogLinearClassifier(nn.Module):

    def __init__(self, input_size, output_size):
        super(LogLinearClassifier, self).__init__()

        # One linear layer - no hidden layers.
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)


if __name__ == '__main__':

    # Creating an instance of LogLinearClassifier.
    model = LogLinearClassifier(input_size=700, output_size=6)

    # Using SGD optimizer.
    optimizer = optim.SGD(model.parameters(), lr=0.005)

    # Training the model
    train_model(model, optimizer)
