import mlx
import numpy as np
import matplotlib.pyplot as plt

class TrainingVisualizer:
    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    def record_metrics(self, train_loss, train_acc, val_loss=None, val_acc=None):
        """Record training and validation metrics for each epoch"""
        self.train_losses.append(float(train_loss))
        self.train_accuracies.append(float(train_acc))
        if val_loss is not None:
            self.val_losses.append(float(val_loss))
        if val_acc is not None:
            self.val_accuracies.append(float(val_acc))
            
    def plot_metrics(self, save_path=None):
        """Plot training and validation metrics"""
        epochs = range(1, len(self.train_losses) + 1)
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        if self.val_losses:
            ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Loss Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        if self.val_accuracies:
            ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Accuracy Over Time')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()

# Modify the MLPMLX class to include training and visualization
class MLPMLX:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.layer1 = LinearLayerMLX(input_dim, hidden_dim)
        self.batch_norm1 = BatchNormalizationMLX(hidden_dim)
        self.tanh = TanhActivationMLX()
        self.layer2 = LinearLayerMLX(hidden_dim, output_dim)
        self.batch_norm2 = BatchNormalizationMLX(output_dim)
        self.cross_entropy_loss = CrossEntropyLossMLX()
        self.visualizer = TrainingVisualizer()

    def forward(self, x, y):
        out = self.layer1.forward(x)
        out = self.batch_norm1.forward(out)
        out = self.tanh.forward(out)
        out = self.layer2.forward(out)
        out = self.batch_norm2.forward(out)
        loss, predictions = self.cross_entropy_loss.forward(out, y)
        return loss, predictions
    
    def calculate_accuracy(self, predictions, targets):
        """Calculate accuracy from predictions and targets"""
        pred_classes = mlx.argmax(predictions, axis=1)
        return mlx.mean(pred_classes == targets)
    
    def train_epoch(self, x_train, y_train, x_val=None, y_val=None, learning_rate=0.01):
        """Train for one epoch and record metrics"""
        # Training
        train_loss, train_predictions = self.forward(x_train, y_train)
        train_accuracy = self.calculate_accuracy(train_predictions, y_train)
        
        # Backpropagation and weight updates
        self.backward()
        self.update_weights(learning_rate)
        
        # Validation
        val_loss = None
        val_accuracy = None
        if x_val is not None and y_val is not None:
            val_loss, val_predictions = self.forward(x_val, y_val)
            val_accuracy = self.calculate_accuracy(val_predictions, y_val)
        
        # Record metrics
        self.visualizer.record_metrics(
            train_loss, 
            train_accuracy,
            val_loss,
            val_accuracy
        )
        
        return train_loss, train_accuracy, val_loss, val_accuracy