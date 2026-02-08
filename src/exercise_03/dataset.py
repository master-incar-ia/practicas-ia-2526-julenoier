import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import Dataset


class NoisyRegressionDataset(Dataset):
    def __init__(self, noise_std=20, size=10000, seed=42):
        np.random.seed(seed)
        self.x = np.random.uniform(0, 100, size=(size,))
        self.delta = np.random.normal(0, noise_std, size=(size,))
        self.y = 100 * np.sin(8 * np.pi * self.x / 100) + 2 + self.delta  
        # Create a DataFrame for visualization
        df = pd.DataFrame(data=np.array([self.x, self.y]).transpose(), columns=["x", "y"])
        self.df = df

        # Reshape for PyTorch compatibility
        self.x = self.x.reshape((-1, 1)).astype(np.float32)
        self.y = self.y.reshape((-1, 1)).astype(np.float32)

        self.x_scale = 100.0 
        self.x_norm = (self.x / self.x_scale).astype(np.float32)

        self.x_tensor = torch.from_numpy(self.x_norm)
        self.y_tensor = torch.from_numpy(self.y)

    def plot(self, filepath):
        ax = sns.scatterplot(self.df, x="x", y="y")
        ax.set_title("Synthetic noisy data of y=100*sin(8*pi*x/100)+2+noise")  
        plt.savefig(filepath)
        plt.show()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x_tensor[idx], self.y_tensor[idx]


if __name__ == "__main__":
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name 
    output_folder.mkdir(exist_ok=True, parents=True)

    dataset = NoisyRegressionDataset()
    print(f"Dataset length: {len(dataset)}")
    print(f"First item: {dataset[0]}")
    # save the plot
    dataset.plot(output_folder / "plot_dataset_example.png")