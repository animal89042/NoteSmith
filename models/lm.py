"""
Language model implementation for NoteSmith.
"""

import torch
import torch.nn as nn


class LanguageModel(nn.Module):
    """
    A language model for text generation and processing.
    """
    
    def __init__(self, vocab_size, embed_size=512, hidden_size=1024, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x, hidden=None):
        """Forward pass through the language model."""
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embedded, hidden)
        output = self.output_layer(output)
        return output, hidden