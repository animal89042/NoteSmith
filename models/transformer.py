"""
Transformer model implementation for NoteSmith.
"""

import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    """
    A transformer-based model for text processing.
    """
    
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt):
        """Forward pass through the transformer."""
        src_emb = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model))
        tgt_emb = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model))
        
        output = self.transformer(src_emb, tgt_emb)
        return self.output_layer(output)