"""
Tokenizer implementation for NoteSmith.
"""

import re
import pickle
from typing import List, Dict, Optional
from collections import Counter


class SimpleTokenizer:
    """
    A simple tokenizer for text processing.
    """
    
    def __init__(self, vocab_size=10000, min_freq=2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.vocab = {}
        self.inverse_vocab = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3,
        }
        
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """Basic tokenization by splitting on whitespace and punctuation."""
        # Simple word tokenization
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        return tokens
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from a list of texts."""
        # Count token frequencies
        token_counter = Counter()
        
        for text in texts:
            preprocessed = self._preprocess_text(text)
            tokens = self._tokenize(preprocessed)
            token_counter.update(tokens)
        
        # Start with special tokens
        self.vocab = self.special_tokens.copy()
        
        # Add most frequent tokens up to vocab_size
        most_common = token_counter.most_common(self.vocab_size - len(self.special_tokens))
        
        for token, freq in most_common:
            if freq >= self.min_freq:
                self.vocab[token] = len(self.vocab)
        
        # Create inverse vocabulary
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token indices."""
        preprocessed = self._preprocess_text(text)
        tokens = self._tokenize(preprocessed)
        
        indices = []
        for token in tokens:
            if token in self.vocab:
                indices.append(self.vocab[token])
            else:
                indices.append(self.vocab['<UNK>'])
        
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """Convert token indices back to text."""
        tokens = []
        for idx in indices:
            if idx in self.inverse_vocab:
                token = self.inverse_vocab[idx]
                if token not in ['<PAD>', '<SOS>', '<EOS>']:
                    tokens.append(token)
        
        return ' '.join(tokens)
    
    def encode_batch(self, texts: List[str], max_length: Optional[int] = None, 
                    add_special_tokens: bool = True) -> List[List[int]]:
        """Encode a batch of texts."""
        encoded_texts = []
        
        for text in texts:
            encoded = self.encode(text)
            
            if add_special_tokens:
                encoded = [self.vocab['<SOS>']] + encoded + [self.vocab['<EOS>']]
            
            if max_length:
                if len(encoded) > max_length:
                    encoded = encoded[:max_length]
                else:
                    encoded += [self.vocab['<PAD>']] * (max_length - len(encoded))
            
            encoded_texts.append(encoded)
        
        return encoded_texts
    
    def save(self, filepath: str):
        """Save tokenizer to file."""
        tokenizer_data = {
            'vocab': self.vocab,
            'inverse_vocab': self.inverse_vocab,
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq,
            'special_tokens': self.special_tokens
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(tokenizer_data, f)
    
    def load(self, filepath: str):
        """Load tokenizer from file."""
        with open(filepath, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        self.vocab = tokenizer_data['vocab']
        self.inverse_vocab = tokenizer_data['inverse_vocab']
        self.vocab_size = tokenizer_data['vocab_size']
        self.min_freq = tokenizer_data['min_freq']
        self.special_tokens = tokenizer_data['special_tokens']
    
    def __len__(self):
        return len(self.vocab)