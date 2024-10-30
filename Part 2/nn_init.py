import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TwoLayerMLP, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)       # First linear layer
        self.bn1 = nn.BatchNorm1d(hidden_dim)             # First batch normalization
        self.fc2 = nn.Linear(hidden_dim, output_dim)      # Second linear layer
        self.bn2 = nn.BatchNorm1d(output_dim)             # Second batch normalization

    def forward(self, x):
        # Forward pass through the first layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.tanh(x)
        
        # Forward pass through the second layer
        x = self.fc2(x)
        x = self.bn2(x)
        return x  # Output is logits, for use with Cross-Entropy Loss

# Example usage:
input_dim = 10   # Dimension of input features
hidden_dim = 20  # Number of neurons in the hidden layer
output_dim = 5   # Number of output classes

model = TwoLayerMLP(input_dim, hidden_dim, output_dim)

# Generate a random input tensor with batch size of 3 and input dimension of 10
inputs = torch.randn(3, input_dim)

# Forward pass
logits = model(inputs)

# Define target labels (for demonstration, using random labels for a 5-class problem)
targets = torch.randint(0, output_dim, (3,))

# Compute the cross-entropy loss
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)

print("Logits:", logits)
print("Loss:", loss.item())
