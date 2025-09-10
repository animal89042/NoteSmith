"""
Training utilities and trainer class for NoteSmith.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


class Trainer:
    """
    Main trainer class for NoteSmith models.
    """
    
    def __init__(self, model, optimizer, criterion, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(device)
        
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
    
    def train(self, train_dataloader, val_dataloader, num_epochs, save_path=None):
        """Full training loop."""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_dataloader)
            val_loss = self.validate(val_dataloader)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            if val_loss < best_val_loss and save_path:
                best_val_loss = val_loss
                self.save_model(save_path)
                print(f'Model saved with validation loss: {val_loss:.4f}')
    
    def save_model(self, path):
        """Save model state dict."""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Load model state dict."""
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)