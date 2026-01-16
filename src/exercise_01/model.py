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
