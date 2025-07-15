import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

from models.mlp import AttentiveMLP

def visualize_attention_scores(model, epoch, save_path="./attention_scores"):
    logging.info(f"Visualizing attention scores for epoch {epoch}")
    model.eval()
    with torch.no_grad():
        # Get attention scores for fc1 and fc2
        fc1_weights = model.fc1.weight.view(-1)
        fc2_weights = model.fc2.weight.view(-1)

        attention_scores_fc1 = model.attention_fc1(fc1_weights).cpu().numpy()
        attention_scores_fc2 = model.attention_fc2(fc2_weights).cpu().numpy()

        # Plotting
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(attention_scores_fc1, bins=50, alpha=0.7, color='blue')
        plt.title(f'Attention Scores for FC1 (Epoch {epoch})')
        plt.xlabel('Attention Score')
        plt.ylabel('Frequency')

        plt.subplot(1, 2, 2)
        plt.hist(attention_scores_fc2, bins=50, alpha=0.7, color='green')
        plt.title(f'Attention Scores for FC2 (Epoch {epoch})')
        plt.xlabel('Attention Score')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(f"{save_path}/epoch_{epoch}_attention_scores.png")
        plt.close()

def train_model(model, train_loader, val_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Make sure the save_path directory exists for attention score visualizations
    if not os.path.exists("./attention_scores"):
        os.makedirs("./attention_scores")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Add regularization loss
            reg_loss = model.regularization_loss()
            total_loss = loss + reg_loss

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Accuracy on validation set: {100 * correct / total}%")

        # Visualize attention scores after each epoch
        visualize_attention_scores(model, epoch)

def get_mnist_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

if __name__ == '__main__':
    # Log start of training
    logging.info("="*50)
    logging.info("Starting Training Session")
    logging.info("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Model configuration
    input_dim = 28 * 28  # MNIST image size
    hidden_dim = 256
    output_dim = 10      # Number of MNIST classes
    lambda_reg = 0.001   # Regularization strength
    epochs = 10
    learning_rate = 0.001
    batch_size = 64
    
    logging.info(f"\nModel Configuration:")
    logging.info(f"Input Dimension: {input_dim}")
    logging.info(f"Hidden Dimension: {hidden_dim}")
    logging.info(f"Output Dimension: {output_dim}")
    logging.info(f"Regularization Lambda: {lambda_reg}")
    logging.info(f"Epochs: {epochs}")
    logging.info(f"Learning Rate: {learning_rate}")
    logging.info(f"Batch Size: {batch_size}")
    
    # Create directories if they don't exist
    os.makedirs("./attention_scores", exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    
    train_loader, val_loader = get_mnist_dataloaders(batch_size)
    
    model = AttentiveMLP(input_dim, hidden_dim, output_dim, lambda_reg).to(device)
    logging.info(f"Model initialized and moved to {device}")
    
    train_model(model, train_loader, val_loader, epochs, learning_rate, device)


