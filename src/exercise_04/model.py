import torch
import torch.nn as nn


# Los modelos en un principio se hacen a huevo. la cantidad de entradas y las salidas las vas inventando

class ConvolutionalNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # conv-conv- maxpool y repetir
        #Primera
        self.convolution = nn.Conv2d(input_dim[0], out_channels = 32, kernel_size=3, stride = 1, padding = 0)
        self.activation = nn.ReLU()
        # Segunda
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=3, stride = 1, padding = 0)
        self.activation2 = nn.ReLU()
        #Maxpool para reducir tamaño
        self.convolution3 = nn.Conv2d(64, 64, kernel_size=3, stride = 1, padding = 0)  # Mantiene 32 canales
        self.activation3 = nn.ReLU()
        
        self.maxpool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.25)
        
        # Dimensión: 32x28x28? No, calculemos bien
        # Entrada 32x32 → conv1(3) → 30x30 → conv2(3) → 28x28 → conv3(3) → 26x26 → pool(2) → 13x13
        # 32 canales * 13 * 13 = 5408
        self.linear = nn.Linear(64*13*13, output_dim)
        
        self.final_activation = nn.Softmax(dim=1) # No se usa 2D porque ya hemos hecho flatten

        
    def forward(self, x, use_activation=True):
        x = self.activation(self.convolution(x))
        x = self.activation2(self.convolution2(x))
        x = self.activation3(self.convolution3(x))
        x = self.maxpool(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        #x2 = self.softmax(x2) Dice deepseek que crossentropy ya tiene softmax
        if use_activation:
            y = self.final_activation(x)
        else:
            y = x
        return y


if __name__ == "__main__":
    pass
    model = ConvolutionalNet((3, 32, 32), 10)
    test_tensor = torch.randn(1, 3, 32, 32)
    output = model(test_tensor, use_activation=False)
    print(f"Salida del modelo (shape): {output.shape}") # Debería ser [1, 10]