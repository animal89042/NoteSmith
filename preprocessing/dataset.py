"""
Dataset classes for NoteSmith.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
from typing import List, Tuple, Optional
import os


class TextDataset(Dataset):
    """
    Basic text dataset for NoteSmith.
    """
    
    def __init__(self, texts: List[str], targets: Optional[List[str]] = None, 
                 tokenizer=None, max_length: int = 512):
        self.texts = texts
        self.targets = targets if targets is not None else texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]
        
        if self.tokenizer:
            # Tokenize input and target
            input_ids = self.tokenizer.encode(text)
            target_ids = self.tokenizer.encode(target)
            
            # Truncate or pad to max_length
            input_ids = self._pad_or_truncate(input_ids)
            target_ids = self._pad_or_truncate(target_ids)
            
            return torch.tensor(input_ids), torch.tensor(target_ids)
        else:
            return text, target
    
    def _pad_or_truncate(self, sequence: List[int]) -> List[int]:
        """Pad or truncate sequence to max_length."""
        if len(sequence) > self.max_length:
            return sequence[:self.max_length]
        else:
            # Assuming tokenizer has <PAD> token at index 0
            padding = [0] * (self.max_length - len(sequence))
            return sequence + padding


class NoteProcessingDataset(Dataset):
    """
    Dataset for note processing tasks (messy notes -> clean sentences).
    """
    
    def __init__(self, messy_notes: List[str], clean_sentences: List[str], 
                 tokenizer=None, max_length: int = 512):
        assert len(messy_notes) == len(clean_sentences), \
            "Number of messy notes and clean sentences must match"
        
        self.messy_notes = messy_notes
        self.clean_sentences = clean_sentences
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.messy_notes)
    
    def __getitem__(self, idx):
        messy = self.messy_notes[idx]
        clean = self.clean_sentences[idx]
        
        if self.tokenizer:
            # Add special tokens
            messy_ids = [self.tokenizer.vocab['<SOS>']] + \
                       self.tokenizer.encode(messy) + \
                       [self.tokenizer.vocab['<EOS>']]
            
            clean_ids = [self.tokenizer.vocab['<SOS>']] + \
                       self.tokenizer.encode(clean) + \
                       [self.tokenizer.vocab['<EOS>']]
            
            # Pad or truncate
            messy_ids = self._pad_or_truncate(messy_ids)
            clean_ids = self._pad_or_truncate(clean_ids)
            
            return torch.tensor(messy_ids), torch.tensor(clean_ids)
        else:
            return messy, clean
    
    def _pad_or_truncate(self, sequence: List[int]) -> List[int]:
        """Pad or truncate sequence to max_length."""
        if len(sequence) > self.max_length:
            return sequence[:self.max_length]
        else:
            padding = [self.tokenizer.vocab['<PAD>']] * (self.max_length - len(sequence))
            return sequence + padding


def load_data_from_csv(filepath: str, text_col: str, target_col: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """Load text data from CSV file."""
    df = pd.read_csv(filepath)
    texts = df[text_col].tolist()
    
    if target_col:
        targets = df[target_col].tolist()
    else:
        targets = texts
    
    return texts, targets


def load_data_from_json(filepath: str) -> Tuple[List[str], List[str]]:
    """Load text data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        # Assume list of strings or list of dicts
        if isinstance(data[0], str):
            return data, data
        elif isinstance(data[0], dict):
            # Assume dict has 'input' and 'target' keys
            inputs = [item.get('input', '') for item in data]
            targets = [item.get('target', item.get('input', '')) for item in data]
            return inputs, targets
    elif isinstance(data, dict):
        # Assume dict has 'inputs' and 'targets' keys
        return data.get('inputs', []), data.get('targets', [])
    
    raise ValueError("Unsupported JSON format")


def create_dataloader(dataset: Dataset, batch_size: int = 32, 
                     shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """Create a DataLoader for the given dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )