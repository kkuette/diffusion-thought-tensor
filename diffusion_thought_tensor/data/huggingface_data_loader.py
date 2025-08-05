"""
Hugging Face Dataset Pipeline for Enhanced Diffusion Thought Tensor Model
Uses real datasets from Hugging Face Hub
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import json
import os
from tqdm import tqdm
import math

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    HF_AVAILABLE = True
except ImportError:
    print(
        "Warning: Hugging Face libraries not available. Install with: pip install datasets transformers"
    )
    HF_AVAILABLE = False


class HuggingFaceDataset(Dataset):
    """Dataset using real Hugging Face datasets"""

    def __init__(
        self,
        dataset_name: str = "roneneldan/TinyStories",
        split: str = "train",
        seq_length: int = 512,
        num_samples: Optional[int] = None,
        cache_dir: str = "data/cache",
        tokenizer_name: str = "gpt2",
    ):
        if not HF_AVAILABLE:
            raise ImportError(
                "Hugging Face libraries required. Install with: pip install datasets transformers"
            )

        self.dataset_name = dataset_name
        self.split = split
        self.seq_length = seq_length
        self.cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)

        # Initialize tokenizer
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vocab_size = len(self.tokenizer)

        # Load and process dataset
        print(f"Loading dataset: {dataset_name} (split: {split})")
        self.processed_data = self._load_and_process_dataset(num_samples)

        print(f"Dataset loaded with {len(self.processed_data)} samples")

    def _load_and_process_dataset(
        self, num_samples: Optional[int]
    ) -> List[torch.Tensor]:
        """Load and tokenize Hugging Face dataset"""

        cache_file = os.path.join(
            self.cache_dir,
            f"hf_{self.dataset_name.replace('/', '_')}_{self.split}_{self.seq_length}_{num_samples or 'all'}.pt",
        )

        # Try cache first
        if os.path.exists(cache_file):
            print(f"Loading cached dataset from {cache_file}")
            return torch.load(cache_file)

        # Load from Hugging Face
        try:
            if self.dataset_name == "roneneldan/TinyStories":
                dataset = load_dataset("roneneldan/TinyStories", split=self.split)
                text_column = "text"
            elif self.dataset_name == "wikitext":
                dataset = load_dataset(
                    "wikitext", "wikitext-2-raw-v1", split=self.split
                )
                text_column = "text"
            elif self.dataset_name.lower() == "openwebtext":
                dataset = load_dataset("Skylion007/openwebtext", name="plain_text", split=self.split, trust_remote_code=True)
                text_column = "text"
            elif self.dataset_name == "bookcorpus":
                dataset = load_dataset("bookcorpus", split=self.split, trust_remote_code=True)
                text_column = "text"
            else:
                # Generic approach - try common column names
                dataset = load_dataset(self.dataset_name, split=self.split)
                text_column = (
                    "text"
                    if "text" in dataset.column_names
                    else dataset.column_names[0]
                )

            print(f"Dataset loaded: {len(dataset)} samples")

        except Exception as e:
            print(f"Error loading dataset {self.dataset_name}: {e}")
            print("Falling back to synthetic data...")
            return self._generate_fallback_data(num_samples or 1000)

        # Limit samples if specified
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))

        # Process and tokenize
        processed_texts = []

        print("Tokenizing texts...")
        for sample in tqdm(dataset):
            text = sample[text_column]

            # Skip only empty texts
            if not text or not text.strip():
                continue

            # Tokenize
            tokens = self.tokenizer(
                text,
                max_length=self.seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = tokens["input_ids"].squeeze(0)  # Remove batch dimension
            processed_texts.append(input_ids)

            # Limit processing for memory
            if len(processed_texts) >= (num_samples or 50000):
                break

        print(f"Processed {len(processed_texts)} texts")

        # Cache the results
        torch.save(processed_texts, cache_file)
        print(f"Dataset cached to {cache_file}")

        return processed_texts

    def _generate_fallback_data(self, num_samples: int) -> List[torch.Tensor]:
        """Generate synthetic data as fallback"""
        print("Generating synthetic fallback data...")

        texts = []
        for _ in range(num_samples):
            # Create more realistic token sequences
            text = []

            # Story-like structure
            story_elements = [
                list(range(100, 150)),  # Characters
                list(range(200, 250)),  # Actions
                list(range(300, 350)),  # Objects
                list(range(400, 450)),  # Locations
                list(range(500, 520)),  # Connectors
            ]

            while len(text) < self.seq_length:
                for element_set in story_elements:
                    if len(text) >= self.seq_length:
                        break
                    token = np.random.choice(element_set)
                    text.append(token)

                    # Add some randomness
                    if np.random.random() < 0.1:
                        text.append(np.random.randint(1000, 2000))

            text = text[: self.seq_length]
            texts.append(torch.tensor(text, dtype=torch.long))

        return texts


    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


def create_huggingface_data_loaders(
    config: Dict,
    dataset_name: str = "roneneldan/TinyStories",
    train_samples: int = 10000,
    eval_samples: int = 1000,
    tokenizer_name: str = "gpt2",
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders using Hugging Face datasets with proper train/val splitting"""

    print(f"Creating Hugging Face data loaders with {dataset_name}...")
    
    # Check if validation split exists for this dataset
    datasets_with_validation = {
        "wikitext": True,
        "roneneldan/TinyStories": False,
        "bookcorpus": False,
        "Skylion007/openwebtext": False,
        "c4": True,
        "EleutherAI/pile": False,
    }
    
    has_validation = datasets_with_validation.get(dataset_name, False)
    
    if has_validation:
        # Use separate validation split
        print(f"Using separate validation split for {dataset_name}")
        
        train_dataset = HuggingFaceDataset(
            dataset_name=dataset_name,
            split="train",
            seq_length=config["model"]["max_seq_length"],
            num_samples=train_samples,
            cache_dir=config.get("data_cache_dir", "data/cache"),
            tokenizer_name=tokenizer_name,
        )
        
        eval_dataset = HuggingFaceDataset(
            dataset_name=dataset_name,
            split="validation",
            seq_length=config["model"]["max_seq_length"],
            num_samples=eval_samples,
            cache_dir=config.get("data_cache_dir", "data/cache"),
            tokenizer_name=tokenizer_name,
        )
    else:
        # Split train data into train/validation
        print(f"📊 Using train/validation split for {dataset_name}")
        
        total_samples = train_samples + eval_samples
        full_dataset = HuggingFaceDataset(
            dataset_name=dataset_name,
            split="train",
            seq_length=config["model"]["max_seq_length"],
            num_samples=total_samples,
            cache_dir=config.get("data_cache_dir", "data/cache"),
            tokenizer_name=tokenizer_name,
        )
        
        # Split the data with proper bounds checking
        actual_size = len(full_dataset)
        print(f"  Dataset actual size: {actual_size}, requested: {total_samples}")
        
        if actual_size < train_samples:
            print(f"⚠️  Warning: Dataset only has {actual_size} samples, using 80/20 split")
            # Use 80/20 split when dataset is smaller than requested
            split_point = int(actual_size * 0.8)
            train_indices = list(range(split_point))
            eval_indices = list(range(split_point, actual_size))
        else:
            # Use requested split
            train_end = min(train_samples, actual_size)
            eval_end = min(train_samples + eval_samples, actual_size)
            train_indices = list(range(train_end))
            eval_indices = list(range(train_end, eval_end))
        
        print(f"  Train indices: {len(train_indices)}, Eval indices: {len(eval_indices)}")
        
        # Ensure we have at least some eval samples
        if len(eval_indices) == 0:
            print(f"⚠️  No eval samples available, using last 10% of train data")
            split_point = max(1, int(len(train_indices) * 0.9))
            eval_indices = train_indices[split_point:]
            train_indices = train_indices[:split_point]
        
        # Create subset datasets with error handling
        try:
            train_data = [full_dataset[i] for i in train_indices]
            eval_data = [full_dataset[i] for i in eval_indices]
        except IndexError as e:
            print(f"❌ Index error: {e}")
            print(f"Dataset size: {len(full_dataset)}, max train idx: {max(train_indices) if train_indices else 'None'}, max eval idx: {max(eval_indices) if eval_indices else 'None'}")
            raise
        
        # Create simple dataset wrappers
        class SubsetDataset(Dataset):
            def __init__(self, data, vocab_size):
                self.data = data
                self.vocab_size = vocab_size
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                if idx >= len(self.data):
                    raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
                return self.data[idx]
        
        train_dataset = SubsetDataset(train_data, full_dataset.vocab_size)
        eval_dataset = SubsetDataset(eval_data, full_dataset.vocab_size)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"].get("dataloader_pin_memory", True),
        drop_last=True,  # Important for consistent batch sizes
        persistent_workers=True if config["hardware"]["num_workers"] > 0 else False,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["hardware"]["num_workers"],
        pin_memory=config["hardware"].get("dataloader_pin_memory", True),
        drop_last=True,
        persistent_workers=True if config["hardware"]["num_workers"] > 0 else False,
    )

    print(f"✅ Hugging Face data loaders created:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Tokenizer: {tokenizer_name}")
    print(f"  Vocabulary size: {train_dataset.vocab_size}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Eval samples: {len(eval_dataset)}")
    print(f"  Sequence length: {config['model']['max_seq_length']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Split method: {'Separate validation split' if has_validation else 'Train data split'}")

    return train_loader, eval_loader


# Supported datasets with split information
SUPPORTED_DATASETS = {
    "tinystories": {
        "name": "roneneldan/TinyStories",
        "has_validation": False,
        "text_column": "text",
        "recommended_samples": 50000,
    },
    "wikitext": {
        "name": "wikitext", 
        "has_validation": True,
        "text_column": "text",
        "recommended_samples": 20000,
    },
    "openwebtext": {
        "name": "Skylion007/openwebtext",
        "has_validation": False, 
        "text_column": "text",
        "recommended_samples": 100000,
    },
    "bookcorpus": {
        "name": "bookcorpus",
        "has_validation": False,
        "text_column": "text", 
        "recommended_samples": 30000,
    },
    "c4": {
        "name": "c4",
        "has_validation": True,
        "text_column": "text",
        "recommended_samples": 50000,
    },
    "pile": {
        "name": "EleutherAI/pile",
        "has_validation": False,
        "text_column": "text",
        "recommended_samples": 100000,
    },
}


if __name__ == "__main__":
    # Test the Hugging Face data loader
    if not HF_AVAILABLE:
        print("Hugging Face libraries not available. Please install with:")
        print("pip install datasets transformers")
        exit(1)

    print("Testing HuggingFaceDataset...")

    test_config = {
        "model": {"max_seq_length": 256, "vocab_size": 50257},
        "training": {"batch_size": 2},
        "hardware": {"num_workers": 0, "dataloader_pin_memory": False},
    }

    # Test with TinyStories
    print("\n=== Testing TinyStories ===")
    try:
        dataset = HuggingFaceDataset(
            dataset_name="roneneldan/TinyStories",
            split="train",
            seq_length=256,
            num_samples=100,
            tokenizer_name="gpt2",
        )

        text = dataset[0]
        print(f"Sample shapes:")
        print(f"  Text: {text.shape}")
        print(f"  Vocabulary size: {dataset.vocab_size}")
        print(f"  Sample text tokens: {text[:10]}")

        # Test data loader
        train_loader, eval_loader = create_huggingface_data_loaders(
            test_config,
            dataset_name="roneneldan/TinyStories",
            train_samples=100,
            eval_samples=20,
        )

        batch_text = next(iter(train_loader))
        print(f"Batch shapes:")
        print(f"  Batch text: {batch_text.shape}")

    except Exception as e:
        print(f"Error testing TinyStories: {e}")

    print("\nHuggingFaceDataset test completed!")
