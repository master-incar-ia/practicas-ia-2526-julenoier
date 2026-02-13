from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torchvision import transforms

from .dataset import CIFAR10Dataset
from .model import MultiLayerPerceptron


def get_device(force: str = "auto") -> torch.device:
    """Return a torch.device based on the `force` option."""
    force = force.lower()
    if force == "cpu":
        return torch.device("cpu")
    if force == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_confusion_matrix(cm, class_names, output_folder):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(12, 10))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, 
                yticklabels=class_names, cmap='Blues', cbar=True)
    plt.title('Confusion Matrix - CIFAR-10 Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_folder / "confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


if __name__ == "__main__":
    output_folder = Path(__file__).parent.parent.parent / "outs" / Path(__file__).parent.name
    output_folder.mkdir(exist_ok=True, parents=True)
    
    # Set the seed for reproducibility
    torch.manual_seed(42)
    
    # Set device
    device = get_device("auto")
    print(f"Using device: {device}")
    
    # Define transformations
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))]
    )
    
    # Load test dataset
    dataset_test = CIFAR10Dataset("./data", train=False, transform=transform)
    
    # Create DataLoader
    test_loader = DataLoader(dataset_test, batch_size=64, shuffle=False)
    
    # Class names for CIFAR-10
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Initialize model
    input_dim = (3, 32, 32)
    output_dim = 10
    model = MultiLayerPerceptron(input_dim, output_dim, num_hidden_neurons=128).to(device)
    
    # Load the best model weights
    best_model_path = output_folder / "best_model.pth"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {best_model_path}")
    
    model.load_state_dict(torch.load(best_model_path, map_location='cpu'))
    print(f"Loaded model from {best_model_path}")
    
    # Evaluate
    model.eval()
    all_predictions = []
    all_targets = []
    
    print("Generating confusion matrix...")
    with torch.no_grad():
        for images, targets in test_loader:
            outputs = model(images, use_activation=False)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.numpy())
            all_targets.extend(targets.numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate accuracy
    accuracy = np.mean(all_predictions == all_targets)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Generate confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Plot and save confusion matrix
    plot_confusion_matrix(cm, class_names, output_folder)
    
    print(f"\nConfusion matrix saved to: {output_folder / 'confusion_matrix.png'}")