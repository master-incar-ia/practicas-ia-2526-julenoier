import torch
import torch.nn as nn


class SimplePerceptron(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.Identity()
        #self.apodo = apodo

    def forward(self, x, use_activation=True):
        x = self.fc(x)
        if use_activation:
            x = self.activation(x)
        return x
    
class MultiplePerceptron(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_neurons):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_neurons)
        self.fc2 = nn.Linear(hidden_neurons, output_dim)
        self.final_activation = nn.Identity()
        #self.apodo = apodo
        self.relu = nn.ReLU()
    def forward(self, x, use_activation=True):
        x = self.fc1(x)
        x = self.relu(x)
        x1 = self.fc2(x)
        x1 = self.relu(x1) 
        if use_activation:
            y = self.final_activation(x1)
        else:
            y = x1
        return y


if __name__ == "__main__":
    model1 = SimplePerceptron(1, 1, "mi_modelo_sencillo")
    model2 = SimplePerceptron(1000, 2, "mi_modelo_desfibrilador")

    x = torch.tensor([1.0])
    print(model1.forward)
    pass
    # print(model)
    # x = torch.tensor([1.0])
    # print(model(x))
    # pass
