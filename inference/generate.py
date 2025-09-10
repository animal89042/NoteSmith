"""
Text generation functionality for NoteSmith.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Union
import random


class TextGenerator:
    """
    Text generator for NoteSmith models.
    """
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def generate_greedy(self, input_text: str, max_length: int = 100) -> str:
        """Generate text using greedy decoding."""
        # Encode input
        input_ids = self.tokenizer.encode(input_text)
        input_tensor = torch.tensor([input_ids]).to(self.device)
        
        generated_ids = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                outputs = self.model(input_tensor)
                
                # Get the most likely next token
                next_token_logits = outputs[0, -1, :]
                next_token_id = torch.argmax(next_token_logits).item()
                
                # Check for end token
                if next_token_id == self.tokenizer.vocab.get('<EOS>', -1):
                    break
                
                # Add to generated sequence
                generated_ids.append(next_token_id)
                
                # Update input for next iteration
                input_tensor = torch.cat([
                    input_tensor, 
                    torch.tensor([[next_token_id]]).to(self.device)
                ], dim=1)
        
        return self.tokenizer.decode(generated_ids)
    
    def generate_sampling(self, input_text: str, max_length: int = 100, 
                         temperature: float = 1.0, top_k: Optional[int] = None,
                         top_p: Optional[float] = None) -> str:
        """Generate text using sampling with temperature, top-k, and top-p."""
        input_ids = self.tokenizer.encode(input_text)
        input_tensor = torch.tensor([input_ids]).to(self.device)
        
        generated_ids = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_tensor)
                next_token_logits = outputs[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k_logits, _ = torch.topk(next_token_logits, top_k)
                    min_top_k = top_k_logits[-1]
                    next_token_logits[next_token_logits < min_top_k] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, 1).item()
                
                # Check for end token
                if next_token_id == self.tokenizer.vocab.get('<EOS>', -1):
                    break
                
                generated_ids.append(next_token_id)
                input_tensor = torch.cat([
                    input_tensor,
                    torch.tensor([[next_token_id]]).to(self.device)
                ], dim=1)
        
        return self.tokenizer.decode(generated_ids)
    
    def generate_beam_search(self, input_text: str, max_length: int = 100,
                           beam_width: int = 5, length_penalty: float = 1.0) -> List[str]:
        """Generate text using beam search."""
        input_ids = self.tokenizer.encode(input_text)
        input_tensor = torch.tensor([input_ids]).to(self.device)
        
        # Initialize beams with input sequence
        beams = [(input_ids, 0.0)]  # (sequence, score)
        
        with torch.no_grad():
            for _ in range(max_length):
                all_candidates = []
                
                for sequence, score in beams:
                    if sequence[-1] == self.tokenizer.vocab.get('<EOS>', -1):
                        all_candidates.append((sequence, score))
                        continue
                    
                    # Get model predictions
                    seq_tensor = torch.tensor([sequence]).to(self.device)
                    outputs = self.model(seq_tensor)
                    next_token_logits = outputs[0, -1, :]
                    log_probs = F.log_softmax(next_token_logits, dim=-1)
                    
                    # Get top-k candidates
                    top_log_probs, top_indices = torch.topk(log_probs, beam_width)
                    
                    for log_prob, token_id in zip(top_log_probs, top_indices):
                        new_sequence = sequence + [token_id.item()]
                        # Apply length penalty
                        length_norm = len(new_sequence) ** length_penalty
                        new_score = (score + log_prob.item()) / length_norm
                        all_candidates.append((new_sequence, new_score))
                
                # Select top beam_width candidates
                all_candidates.sort(key=lambda x: x[1], reverse=True)
                beams = all_candidates[:beam_width]
                
                # Check if all beams ended
                if all(seq[-1] == self.tokenizer.vocab.get('<EOS>', -1) for seq, _ in beams):
                    break
        
        # Return decoded sequences
        results = []
        for sequence, _ in beams:
            decoded = self.tokenizer.decode(sequence)
            results.append(decoded)
        
        return results
    
    def clean_notes(self, messy_notes: Union[str, List[str]], 
                   method: str = 'greedy', **kwargs) -> Union[str, List[str]]:
        """Clean messy notes into proper sentences."""
        if isinstance(messy_notes, str):
            if method == 'greedy':
                return self.generate_greedy(messy_notes, **kwargs)
            elif method == 'sampling':
                return self.generate_sampling(messy_notes, **kwargs)
            elif method == 'beam':
                results = self.generate_beam_search(messy_notes, **kwargs)
                return results[0] if results else messy_notes
        else:
            # Process batch of notes
            cleaned_notes = []
            for note in messy_notes:
                cleaned = self.clean_notes(note, method=method, **kwargs)
                cleaned_notes.append(cleaned)
            return cleaned_notes