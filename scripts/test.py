import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from model import FashionCNN
import matplotlib.pyplot as plt
import numpy as np

# ============= IMPORTANT FOR WINDOWS =============
if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = torchvision.datasets.FashionMNIST(
        root='../data',
        train=False,
        download=True,
        transform=transform
    )

    # FIXED: num_workers=0 for Windows
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0  # Changed from 2 to 0 for Windows
    )

    print(f"Test samples: {len(test_dataset)}")

    # Load trained model
    model = FashionCNN().to(device)
    model.load_state_dict(torch.load('../models/fashion_mnist_cnn.pth'))
    model.eval()
    print("Model loaded successfully!")

    # Test the model
    print("\nEvaluating model on test set...")
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Overall accuracy
    overall_accuracy = 100 * correct / total
    print(f'\nOverall Test Accuracy: {overall_accuracy:.2f}%')

    # Per-class accuracy
    print('\nPer-class Accuracy:')
    for i in range(10):
        class_acc = 100 * class_correct[i] / class_total[i]
        print(f'{class_names[i]:15s}: {class_acc:.2f}%')

    # Visualize predictions
    def visualize_predictions(num_images=15):
        model.eval()
        images, labels = next(iter(test_loader))
        images, labels = images.to(device), labels.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
        
        images = images.cpu()
        labels = labels.cpu()
        predicted = predicted.cpu()
        
        fig, axes = plt.subplots(3, 5, figsize=(12, 8))
        for idx, ax in enumerate(axes.flat):
            if idx < num_images:
                img = images[idx].squeeze()
                true_label = class_names[labels[idx]]
                pred_label = class_names[predicted[idx]]
                
                ax.imshow(img, cmap='gray')
                color = 'green' if labels[idx] == predicted[idx] else 'red'
                ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color)
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('../models/test_predictions.png')
        print('\nPrediction visualizations saved to ../models/test_predictions.png')
        
        # Center the plot window on the screen
        try:
            mng = plt.get_current_fig_manager()
            if hasattr(mng, 'window'):
                root = mng.window
                root.update_idletasks()
                screen_width = root.winfo_screenwidth()
                screen_height = root.winfo_screenheight()
                window_width = root.winfo_width()
                window_height = root.winfo_height()
                position_x = (screen_width // 2) - (window_width // 2)
                position_y = (screen_height // 2) - (window_height // 2)
                root.geometry(f'+{position_x}+{position_y}')
        except Exception:
            pass # Ignore if centering fails

        plt.show()

    visualize_predictions()