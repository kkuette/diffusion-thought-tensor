#!/usr/bin/env python3
"""
Test script to debug generation issues
"""

import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from diffusion_thought_tensor.model.masked_fast_model import MaskedFastDiffusionModel, MaskingStrategy
from transformers import AutoTokenizer

def test_generation():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model with same config as interactive chat
    masking_strategy = MaskingStrategy(
        mask_ratio=0.8,
        confidence_threshold=0.7,
        progressive_unmasking=True
    )
    
    model = MaskedFastDiffusionModel(
        vocab_size=50257,
        embed_dim=384,
        num_layers=6,
        num_heads=6,
        max_seq_length=2048,
        max_new_tokens=1024,
        thought_dims=(16, 16, 8),
        thought_stack_size=8,
        dropout=0.2,
        masking_strategy=masking_strategy
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Try loading checkpoint
    checkpoint_path = "./diffusion_thought_tensor/outputs/masked_checkpoints/best_masked_model.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Checkpoint loaded")
    else:
        print("⚠️ No checkpoint found, using random weights")
    
    # Test with simple prompt
    test_prompt = "Hello, how are you?"
    print(f"\nTest prompt: '{test_prompt}'")
    
    # Tokenize
    tokens = tokenizer(test_prompt, return_tensors="pt")["input_ids"].to(device)
    print(f"Tokenized shape: {tokens.shape}")
    print(f"Token IDs: {tokens[0].tolist()}")
    
    # Generate
    batch_size = tokens.shape[0]
    thought_stack = model.initialize_thought_stack(batch_size, device)
    
    generated_tokens, _, stats = model.masked_generate(
        initial_stack=thought_stack,
        prompt_tokens=tokens,
        num_steps=25,
        temperature=0.8
    )
    
    print(f"\nGenerated shape: {generated_tokens.shape}")
    print(f"Generation stats: {stats}")
    
    # Decode
    response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print(f"\nFull response: '{response}'")
    
    # Extract only the new part
    prompt_len = tokens.shape[1]
    if generated_tokens.shape[1] > prompt_len:
        new_tokens = generated_tokens[0, prompt_len:]
        new_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"Generated text only: '{new_response}'")
    
    # Check token distribution
    unique_tokens = torch.unique(generated_tokens)
    print(f"\nUnique tokens generated: {len(unique_tokens)}")
    print(f"Token range: {unique_tokens.min().item()} - {unique_tokens.max().item()}")

if __name__ == "__main__":
    test_generation()