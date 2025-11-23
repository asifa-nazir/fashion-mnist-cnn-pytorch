import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from model import FashionCNN
import matplotlib.pyplot as plt
import os

# ============= IMPORTANT FOR WINDOWS =============
if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20

    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load Fashion-MNIST dataset
    print("Loading Fashion-MNIST dataset...")
    train_dataset = torchvision.datasets.FashionMNIST(
        root='../data',
        train=True,
        download=True,
        transform=transform
    )

    # FIXED: num_workers=0 for Windows
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Changed from 2 to 0 for Windows
    )

    print(f"Training samples: {len(train_dataset)}")

    # Initialize model
    model = FashionCNN().to(device)
    print(f"\nModel architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    train_losses = []
    train_accuracies = []

    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress
            if (i + 1) % 200 == 0:
                print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

    # Save model
    output_dir = '../models'
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, 'fashion_mnist_cnn.pth')
    torch.save(model.state_dict(), model_path)
    print(f'\nModel saved to {model_path}')

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1) 
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    plt.tight_layout()
    curves_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(curves_path)
    print(f'Training curves saved to {curves_path}')
    plt.show()