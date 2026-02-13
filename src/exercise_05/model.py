import torch
import torch.nn as nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_neurons, apodo=None):
        super().__init__()
        self.flat = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(input_dim, num_hidden_neurons)
        self.fc2 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
        self.fc3 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
        self.fc4 = nn.Linear(num_hidden_neurons, output_dim)  

        self.relu = nn.ReLU()
        self.out_act = nn.Identity()
        self.apodo = apodo

    def forward(self, x, use_activation=True):
        x = self.flat(x) # Es necesario obtener un vector a partir de la imagen porque las FC no trabajan con imagenes
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        if use_activation:
            x = self.out_act(x)
        return x

if __name__ == "__main__":
    model = MultiLayerPerceptron(1000, 2, 256)

    x = torch.randn(1, 1000)
    print(model.forward(x))
    pass