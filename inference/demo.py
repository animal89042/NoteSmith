"""
Demo script for NoteSmith - interactive note cleaning.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.transformer import TransformerModel
from models.lm import LanguageModel
from models.seq2seq import Seq2SeqModel
from preprocessing.tokenizer import SimpleTokenizer
from inference.generate import TextGenerator
import torch


class NoteSmithDemo:
    """
    Interactive demo for NoteSmith.
    """
    
    def __init__(self, model_type='lm', model_path=None, tokenizer_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        self.tokenizer = SimpleTokenizer()
        if tokenizer_path and os.path.exists(tokenizer_path):
            self.tokenizer.load(tokenizer_path)
        else:
            # Use a simple vocabulary for demo
            self._init_demo_tokenizer()
        
        # Initialize model
        self.model = self._init_model(model_type)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Initialize generator
        self.generator = TextGenerator(self.model, self.tokenizer, self.device)
        
        print(f"NoteSmith Demo initialized!")
        print(f"Model type: {model_type}")
        print(f"Device: {self.device}")
        print(f"Vocabulary size: {len(self.tokenizer)}")
    
    def _init_demo_tokenizer(self):
        """Initialize a simple demo tokenizer."""
        # Create a basic vocabulary for demo purposes
        demo_texts = [
            "hello world",
            "this is a test",
            "clean up messy notes",
            "convert text to sentences",
            "machine learning model"
        ]
        self.tokenizer.build_vocab(demo_texts)
    
    def _init_model(self, model_type):
        """Initialize the specified model."""
        vocab_size = len(self.tokenizer) if len(self.tokenizer) > 0 else 1000
        
        if model_type == 'transformer':
            return TransformerModel(vocab_size)
        elif model_type == 'lm':
            return LanguageModel(vocab_size)
        elif model_type == 'seq2seq':
            return Seq2SeqModel(vocab_size, vocab_size, hidden_size=256)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def clean_note(self, messy_note, method='greedy', **kwargs):
        """Clean a messy note."""
        try:
            if method == 'greedy':
                return self.generator.generate_greedy(messy_note, **kwargs)
            elif method == 'sampling':
                return self.generator.generate_sampling(messy_note, **kwargs)
            elif method == 'beam':
                results = self.generator.generate_beam_search(messy_note, **kwargs)
                return results[0] if results else messy_note
            else:
                return f"Unknown method: {method}"
        except Exception as e:
            return f"Error processing note: {str(e)}"
    
    def interactive_demo(self):
        """Run interactive demo."""
        print("\n" + "="*50)
        print("Welcome to NoteSmith Interactive Demo!")
        print("Type 'quit' to exit, 'help' for commands")
        print("="*50 + "\n")
        
        while True:
            try:
                user_input = input("Enter messy note: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Thanks for using NoteSmith!")
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif not user_input:
                    continue
                
                # Process the note
                print("\nProcessing...")
                
                # Try different methods
                methods = ['greedy', 'sampling']
                for method in methods:
                    try:
                        cleaned = self.clean_note(user_input, method=method, max_length=50)
                        print(f"{method.capitalize()}: {cleaned}")
                    except Exception as e:
                        print(f"{method.capitalize()}: Error - {str(e)}")
                
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def _show_help(self):
        """Show help information."""
        print("\nNoteSmith Demo Commands:")
        print("- Type any text to clean it up")
        print("- 'help' - Show this help message")
        print("- 'quit' - Exit the demo")
        print("\nExample inputs:")
        print("- 'mtg tmrw 3pm discuss proj'")
        print("- 'buy milk eggs bread 2day'")
        print("- 'call john re meeting setup'")
        print()
    
    def batch_demo(self, notes):
        """Demonstrate batch processing."""
        print("Batch Processing Demo:")
        print("="*30)
        
        for i, note in enumerate(notes, 1):
            print(f"\nNote {i}: {note}")
            cleaned = self.clean_note(note)
            print(f"Clean: {cleaned}")


def main():
    """Main demo function."""
    # Sample messy notes for demo
    sample_notes = [
        "mtg tmrw 3pm",
        "buy milk eggs bread",
        "call john re project",
        "dentist appt tue 2pm",
        "finish report by fri"
    ]
    
    try:
        # Initialize demo
        demo = NoteSmithDemo(model_type='lm')
        
        # Show batch demo first
        demo.batch_demo(sample_notes)
        
        # Run interactive demo
        demo.interactive_demo()
        
    except Exception as e:
        print(f"Demo error: {str(e)}")
        print("Note: This is a demo with an untrained model.")
        print("For real functionality, train the model first!")


if __name__ == "__main__":
    main()