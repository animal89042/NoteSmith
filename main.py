"""
NoteSmith - Turning messy notes into clean sentences.

Main entry point for the NoteSmith application.
"""

import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transformer import TransformerModel
from models.lm import LanguageModel
from models.seq2seq import Seq2SeqModel
from preprocessing.tokenizer import SimpleTokenizer
from preprocessing.dataset import NoteProcessingDataset, load_data_from_csv, load_data_from_json
from training.trainer import Trainer
from training.utils import set_seed, get_device, count_parameters
from inference.generate import TextGenerator
from inference.demo import NoteSmithDemo
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='NoteSmith - Clean up messy notes')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--data', required=True, help='Path to training data')
    train_parser.add_argument('--model', default='lm', choices=['lm', 'transformer', 'seq2seq'],
                            help='Model type to train')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--save-path', default='./results/model.pth', help='Path to save model')
    train_parser.add_argument('--tokenizer-path', default='./results/tokenizer.pkl', help='Path to save tokenizer')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate clean text from messy notes')
    gen_parser.add_argument('--model-path', required=True, help='Path to trained model')
    gen_parser.add_argument('--tokenizer-path', required=True, help='Path to tokenizer')
    gen_parser.add_argument('--input', required=True, help='Input messy note')
    gen_parser.add_argument('--method', default='greedy', choices=['greedy', 'sampling', 'beam'],
                          help='Generation method')
    gen_parser.add_argument('--max-length', type=int, default=100, help='Maximum generation length')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run interactive demo')
    demo_parser.add_argument('--model-type', default='lm', choices=['lm', 'transformer', 'seq2seq'],
                           help='Model type for demo')
    demo_parser.add_argument('--model-path', help='Path to trained model (optional)')
    demo_parser.add_argument('--tokenizer-path', help='Path to tokenizer (optional)')
    
    return parser.parse_args()


def create_model(model_type, vocab_size):
    """Create a model of the specified type."""
    if model_type == 'lm':
        return LanguageModel(vocab_size, embed_size=512, hidden_size=1024)
    elif model_type == 'transformer':
        return TransformerModel(vocab_size, d_model=512, nhead=8, num_layers=6)
    elif model_type == 'seq2seq':
        return Seq2SeqModel(vocab_size, vocab_size, hidden_size=512)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_training_data(data_path):
    """Load training data from file."""
    if data_path.endswith('.csv'):
        # Assume CSV has columns 'messy' and 'clean'
        return load_data_from_csv(data_path, 'messy', 'clean')
    elif data_path.endswith('.json'):
        return load_data_from_json(data_path)
    else:
        # Try to load as text file with lines alternating between messy and clean
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        messy_notes = lines[::2]  # Even indices
        clean_sentences = lines[1::2]  # Odd indices
        
        if len(messy_notes) != len(clean_sentences):
            raise ValueError("Data file should have alternating messy/clean lines")
        
        return messy_notes, clean_sentences


def train_model(args):
    """Train a NoteSmith model."""
    print("Starting training...")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    try:
        messy_notes, clean_sentences = load_training_data(args.data)
        print(f"Loaded {len(messy_notes)} training examples")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=10000)
    all_texts = messy_notes + clean_sentences
    tokenizer.build_vocab(all_texts)
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Save tokenizer
    os.makedirs(os.path.dirname(args.tokenizer_path), exist_ok=True)
    tokenizer.save(args.tokenizer_path)
    print(f"Tokenizer saved to {args.tokenizer_path}")
    
    # Create model
    model = create_model(args.model, len(tokenizer))
    param_count = count_parameters(model)
    print(f"Model created with {param_count:,} parameters")
    
    # Create dataset and dataloader
    dataset = NoteProcessingDataset(messy_notes, clean_sentences, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab['<PAD>'])
    trainer = Trainer(model, optimizer, criterion, device)
    
    # Train
    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    trainer.save_model(args.save_path)
    print(f"Model saved to {args.save_path}")


def generate_text(args):
    """Generate clean text from messy notes."""
    print("Loading model for generation...")
    
    # Load tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.load(args.tokenizer_path)
    
    # Load model
    model = create_model('lm', len(tokenizer))  # Assume language model for now
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    
    # Create generator
    generator = TextGenerator(model, tokenizer)
    
    # Generate
    if args.method == 'greedy':
        result = generator.generate_greedy(args.input, max_length=args.max_length)
    elif args.method == 'sampling':
        result = generator.generate_sampling(args.input, max_length=args.max_length)
    elif args.method == 'beam':
        results = generator.generate_beam_search(args.input, max_length=args.max_length)
        result = results[0] if results else args.input
    
    print(f"Input: {args.input}")
    print(f"Output: {result}")


def run_demo(args):
    """Run interactive demo."""
    demo = NoteSmithDemo(
        model_type=args.model_type,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path
    )
    demo.interactive_demo()


def main():
    """Main function."""
    args = parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'generate':
        generate_text(args)
    elif args.command == 'demo':
        run_demo(args)
    else:
        print("Please specify a command: train, generate, or demo")
        print("Use --help for more information")


if __name__ == "__main__":
    main()