"""
Sequence-to-sequence model implementation for NoteSmith.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encoder for sequence-to-sequence model."""
    
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        
    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class Decoder(nn.Module):
    """Decoder for sequence-to-sequence model."""
    
    def __init__(self, output_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_token, hidden):
        output = self.embedding(input_token).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        output = self.out(output[0])
        return output, hidden


class Seq2SeqModel(nn.Module):
    """Complete sequence-to-sequence model."""
    
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_size)
        self.decoder = Decoder(output_size, hidden_size)
        
    def forward(self, input_seq, target_seq=None):
        encoder_outputs, encoder_hidden = self.encoder(input_seq)
        
        if target_seq is not None:
            # Training mode
            decoder_outputs = []
            decoder_hidden = encoder_hidden
            
            for i in range(target_seq.size(0)):
                decoder_output, decoder_hidden = self.decoder(target_seq[i], decoder_hidden)
                decoder_outputs.append(decoder_output)
                
            return torch.stack(decoder_outputs)
        else:
            # Inference mode
            return encoder_hidden