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
    
class MultiLayerPerceptron_2(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.flat = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(3072, 512)
        self.BatchNorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.2) # Apaga neuronas
        self.linear2 = nn.Linear(512, 256) # Dimensiones reducidas en comparación con Linear
        self.BatchNorm2 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, 128) # Reducción de dimensiones
        self.BatchNorm3 = nn.BatchNorm1d(128)
        self.output_layer = nn.Linear(128, output_dim)  
        self.activation = nn.ReLU() # Capa de activación


    def forward(self, x):
        x = self.flat(x) # Es necesario obtener un vector a partir de la imagen porque las FC no trabajan con imagenes
        
        x = self.linear(x)
        x = self.activation(x)
        x = self.BatchNorm(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.activation(x)
        x = self.BatchNorm2(x)
        x = self.dropout(x)

        x = self.linear3(x)
        x = self.activation(x)
        x = self.BatchNorm3(x)
        x = self.dropout(x)

        x = self.output_layer(x)
        
        return x


if __name__ == "__main__":
    model = MultiLayerPerceptron(1000, 2, 256)
    x = torch.randn(1, 1000)
    print(model.forward(x))
    pass