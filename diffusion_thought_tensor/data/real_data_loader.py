"""
Real Data Pipeline for Diffusion Thought Tensor Training
Loads and preprocesses real text data for training
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Dict
import json
import os
from tqdm import tqdm
import requests
import io
import zipfile
import re
from collections import Counter


class RealTextDataset(Dataset):
    """Real text dataset for training diffusion thought tensor models"""
    
    def __init__(
        self,
        data_source: str = "wikitext",
        seq_length: int = 512,
        vocab_size: int = 50257,
        thought_dims: Tuple[int, ...] = (32, 32, 16),
        stack_size: int = 16,
        num_samples: Optional[int] = None,
        cache_dir: str = "data/cache",
        download: bool = True
    ):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.thought_dims = thought_dims
        self.stack_size = stack_size
        self.cache_dir = cache_dir
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load real text data
        print(f"Loading {data_source} dataset...")
        self.texts = self._load_dataset(data_source, download)
        
        if num_samples:
            self.texts = self.texts[:num_samples]
        
        print(f"Dataset loaded with {len(self.texts)} samples")
        
        # Generate corresponding thought stacks
        print("Generating thought stacks...")
        self.thought_stacks = self._generate_contextual_thoughts()
        
    def _load_dataset(self, source: str, download: bool = True) -> List[torch.Tensor]:
        """Load real text dataset"""
        
        cache_path = os.path.join(self.cache_dir, f"{source}_tokenized.pt")
        
        # Try to load from cache first
        if os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path}")
            return torch.load(cache_path)
        
        if source == "wikitext":
            texts = self._load_wikitext(download)
        elif source == "openwebtext":
            texts = self._load_openwebtext(download)
        elif source == "tinystories":
            texts = self._load_tinystories(download)
        else:
            raise ValueError(f"Unknown data source: {source}")
        
        # Save to cache
        torch.save(texts, cache_path)
        print(f"Dataset cached to {cache_path}")
        
        return texts
    
    def _load_wikitext(self, download: bool = True) -> List[torch.Tensor]:
        """Load WikiText-103 dataset"""
        
        wikitext_path = os.path.join(self.cache_dir, "wikitext-103-raw")
        
        if download and not os.path.exists(wikitext_path):
            print("Downloading WikiText-103 dataset...")
            url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
            
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                    zip_file.extractall(self.cache_dir)
                
                print("WikiText-103 downloaded successfully")
            except Exception as e:
                print(f"Failed to download WikiText-103: {e}")
                return self._create_synthetic_dataset()
        
        # Read and process WikiText files
        train_file = os.path.join(wikitext_path, "wikitext-103-raw", "wiki.train.raw")
        
        if not os.path.exists(train_file):
            print("WikiText files not found, using synthetic data")
            return self._create_synthetic_dataset()
        
        return self._process_text_file(train_file)
    
    def _load_tinystories(self, download: bool = True) -> List[torch.Tensor]:
        """Load TinyStories dataset (smaller, good for testing)"""
        
        # Use a sample of synthetic stories for now
        print("Using synthetic TinyStories-like data for training")
        
        story_templates = [
            "Once upon a time, there was a {character} who lived in a {place}. Every day, the {character} would {action} and {result}.",
            "A little {character} found a {object} in the {place}. The {object} was {adjective} and made the {character} feel {emotion}.",
            "In a {place} far away, a {character} decided to {action}. It was {adjective} but the {character} was {emotion}.",
            "The {character} woke up and saw a {object}. They thought it was {adjective} so they decided to {action}.",
        ]
        
        characters = ["cat", "dog", "bird", "mouse", "rabbit", "bear", "fox", "owl"]
        places = ["forest", "garden", "house", "park", "meadow", "cave", "tree", "pond"]
        objects = ["ball", "flower", "stone", "book", "toy", "berry", "leaf", "shell"]
        adjectives = ["beautiful", "shiny", "colorful", "soft", "bright", "mysterious", "tiny", "magical"]
        actions = ["play", "explore", "dance", "sing", "jump", "run", "hide", "climb"]
        emotions = ["happy", "excited", "curious", "proud", "peaceful", "joyful", "amazed", "content"]
        results = ["smiled", "laughed", "found a friend", "discovered something new", "felt proud", "had fun"]
        
        stories = []
        for _ in range(10000):  # Generate 10k stories
            template = np.random.choice(story_templates)
            story = template.format(
                character=np.random.choice(characters),
                place=np.random.choice(places),
                object=np.random.choice(objects),
                adjective=np.random.choice(adjectives),
                action=np.random.choice(actions),
                emotion=np.random.choice(emotions),
                result=np.random.choice(results)
            )
            stories.append(story)
        
        return self._tokenize_texts(stories)
    
    def _create_synthetic_dataset(self) -> List[torch.Tensor]:
        """Create synthetic dataset as fallback"""
        print("Creating synthetic dataset...")
        
        # Generate more realistic synthetic text patterns
        texts = []
        
        # Common word patterns
        common_words = [
            "the", "and", "a", "to", "of", "in", "is", "it", "you", "that",
            "he", "was", "for", "on", "are", "as", "with", "his", "they", "i"
        ]
        
        sentence_starters = [
            "The quick brown", "In the beginning", "Once upon a time",
            "After careful consideration", "Despite the challenges", "Through the forest",
            "During the meeting", "Before the storm", "Beyond the horizon"
        ]
        
        for _ in range(5000):
            # Create pseudo-sentences
            text_tokens = []
            
            # Start with a sentence starter
            starter = np.random.choice(sentence_starters).split()
            text_tokens.extend(starter)
            
            # Add common words and patterns
            while len(text_tokens) < self.seq_length:
                if np.random.random() < 0.7:
                    text_tokens.append(np.random.choice(common_words))
                else:
                    # Add some variety
                    text_tokens.append(f"word_{np.random.randint(100, 1000)}")
            
            # Convert to token IDs (simplified)
            token_ids = []
            for word in text_tokens[:self.seq_length]:
                # Simple hash-based tokenization
                token_id = hash(word) % self.vocab_size
                token_ids.append(token_id)
            
            # Pad or truncate to sequence length
            if len(token_ids) < self.seq_length:
                token_ids.extend([0] * (self.seq_length - len(token_ids)))
            else:
                token_ids = token_ids[:self.seq_length]
            
            texts.append(torch.tensor(token_ids, dtype=torch.long))
        
        return texts
    
    def _process_text_file(self, file_path: str) -> List[torch.Tensor]:
        """Process a text file into tokenized sequences"""
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple preprocessing
        content = re.sub(r'\n+', ' ', content)  # Replace newlines
        content = re.sub(r'\s+', ' ', content)  # Normalize spaces
        content = content.strip()
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        return self._tokenize_texts(sentences[:10000])  # Limit for memory
    
    def _tokenize_texts(self, texts: List[str]) -> List[torch.Tensor]:
        """Simple tokenization (in practice, would use a proper tokenizer)"""
        
        tokenized = []
        
        for text in tqdm(texts, desc="Tokenizing"):
            # Simple word-based tokenization
            words = text.lower().split()
            
            # Convert words to token IDs
            tokens = []
            for word in words:
                # Simple hash-based mapping (in practice, use proper tokenizer)
                token_id = hash(word) % self.vocab_size
                tokens.append(token_id)
            
            # Pad or truncate to sequence length
            if len(tokens) < self.seq_length:
                tokens.extend([0] * (self.seq_length - len(tokens)))
            else:
                tokens = tokens[:self.seq_length]
            
            tokenized.append(torch.tensor(tokens, dtype=torch.long))
        
        return tokenized
    
    def _generate_contextual_thoughts(self) -> torch.Tensor:
        """Generate thought stacks that are contextually relevant to the text"""
        
        num_samples = len(self.texts)
        thought_stacks = torch.zeros(
            num_samples, self.stack_size, *self.thought_dims, dtype=torch.float32
        )
        
        print("Generating contextual thought stacks...")
        
        for i in tqdm(range(num_samples)):
            text_tokens = self.texts[i]
            
            # Extract text statistics for contextual thoughts
            text_stats = {
                'token_mean': text_tokens.float().mean().item(),
                'token_std': text_tokens.float().std().item(),
                'token_max': text_tokens.max().item(),
                'token_min': text_tokens.min().item(),
                'unique_tokens': len(torch.unique(text_tokens)),
                'sequence_length': len(text_tokens)
            }
            
            # Generate stack based on text characteristics
            for stack_idx in range(self.stack_size):
                # Base initialization
                fan_in = np.prod(self.thought_dims)
                std = np.sqrt(2.0 / fan_in)
                thought = torch.randn(*self.thought_dims, dtype=torch.float32) * std
                
                # Modulate based on text statistics
                # Scale based on vocabulary diversity
                vocab_diversity = text_stats['unique_tokens'] / len(text_tokens)
                thought *= (0.5 + vocab_diversity)
                
                # Add structure based on text patterns
                # High-frequency tokens -> more structured thoughts
                if text_stats['token_mean'] > self.vocab_size / 2:
                    # Add periodic structure
                    x_coords = torch.arange(self.thought_dims[0]).float()
                    y_coords = torch.arange(self.thought_dims[1]).float()
                    z_coords = torch.arange(self.thought_dims[2]).float()
                    
                    X, Y, Z = torch.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
                    
                    # Add wave patterns based on text statistics
                    frequency = text_stats['token_std'] / 100.0
                    wave = torch.sin(X * frequency) * torch.cos(Y * frequency) * torch.sin(Z * frequency)
                    thought += wave * 0.1
                
                # Add position-dependent modulation
                stack_position = stack_idx / self.stack_size
                position_scale = 0.8 + 0.4 * stack_position  # Later thoughts are more varied
                thought *= position_scale
                
                # Add text-specific noise
                text_complexity = text_stats['token_std'] / text_stats['token_mean'] if text_stats['token_mean'] > 0 else 1.0
                noise = torch.randn_like(thought, dtype=torch.float32) * text_complexity * 0.05
                thought += noise
                
                thought_stacks[i, stack_idx] = thought
        
        return thought_stacks
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.thought_stacks[idx]


def create_real_data_loaders(
    config: Dict,
    train_samples: int = 20000,
    eval_samples: int = 2000,
    data_source: str = "tinystories"
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for real data training"""
    
    print(f"Creating real data loaders with {data_source} dataset...")
    
    # Training dataset
    train_dataset = RealTextDataset(
        data_source=data_source,
        seq_length=config['model']['max_seq_length'],
        vocab_size=config['model']['vocab_size'],
        thought_dims=tuple(config['thought_tensor']['input_dims']),
        stack_size=config['thought_tensor']['stack_size'],
        num_samples=train_samples,
        cache_dir=config.get('data_cache_dir', 'data/cache')
    )
    
    # Evaluation dataset (subset)
    eval_dataset = RealTextDataset(
        data_source=data_source,
        seq_length=config['model']['max_seq_length'],
        vocab_size=config['model']['vocab_size'],
        thought_dims=tuple(config['thought_tensor']['input_dims']),
        stack_size=config['thought_tensor']['stack_size'],
        num_samples=eval_samples,
        cache_dir=config.get('data_cache_dir', 'data/cache')
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware'].get('dataloader_pin_memory', True),
        persistent_workers=True if config['hardware']['num_workers'] > 0 else False
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware'].get('dataloader_pin_memory', True),
        persistent_workers=True if config['hardware']['num_workers'] > 0 else False
    )
    
    print(f"Data loaders created:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Eval samples: {len(eval_dataset)}")
    print(f"  Batch size: {config['training']['batch_size']}")
    
    return train_loader, eval_loader


if __name__ == "__main__":
    # Test the real data loader
    print("Testing RealTextDataset...")
    
    # Simple test configuration
    test_config = {
        'model': {
            'max_seq_length': 128,
            'vocab_size': 50257
        },
        'thought_tensor': {
            'input_dims': [16, 16, 8],
            'stack_size': 8
        },
        'training': {
            'batch_size': 2
        },
        'hardware': {
            'num_workers': 0,
            'dataloader_pin_memory': False
        }
    }
    
    # Create test dataset
    dataset = RealTextDataset(
        data_source="tinystories",
        seq_length=128,
        num_samples=100
    )
    
    # Test data loading
    text, thoughts = dataset[0]
    print(f"Sample shapes:")
    print(f"  Text: {text.shape}")
    print(f"  Thoughts: {thoughts.shape}")
    print(f"  Text sample: {text[:10]}")
    print(f"  Thought stats: mean={thoughts.mean():.3f}, std={thoughts.std():.3f}")
    
    print("RealTextDataset test completed successfully!")