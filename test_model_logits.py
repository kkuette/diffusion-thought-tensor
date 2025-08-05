#!/usr/bin/env python3
"""
Test model logits to debug generation issues
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from diffusion_thought_tensor.model.masked_fast_model import MaskedFastDiffusionModel, MaskingStrategy

def test_model_forward():
    # Create model with same config
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
    model.eval()
    
    # Load checkpoint
    checkpoint_path = "./diffusion_thought_tensor/outputs/masked_checkpoints/best_masked_model.pt"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("✅ Checkpoint loaded")
    
    # Test forward pass
    batch_size = 1
    seq_length = 20
    
    # Create test input
    test_tokens = torch.randint(0, 1000, (batch_size, seq_length), device=device)
    thought_stack = model.initialize_thought_stack(batch_size, device)
    timestep = torch.zeros(batch_size, device=device)
    
    print(f"\nTest input shape: {test_tokens.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits, new_thought = model(test_tokens, thought_stack, timestep)
    
    print(f"\nOutput shapes:")
    print(f"  Logits: {logits.shape}")
    print(f"  New thought: {new_thought.shape}")
    
    # Analyze logits
    print(f"\nLogits statistics:")
    print(f"  Mean: {logits.mean().item():.4f}")
    print(f"  Std: {logits.std().item():.4f}")
    print(f"  Min: {logits.min().item():.4f}")
    print(f"  Max: {logits.max().item():.4f}")
    
    # Check probability distribution
    probs = F.softmax(logits[0, 0], dim=-1)  # First token probabilities
    top_probs, top_indices = torch.topk(probs, 10)
    
    print(f"\nTop 10 token probabilities (first position):")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        print(f"  Token {idx.item()}: {prob.item():.4f}")
    
    # Check entropy
    entropy = -(probs * torch.log(probs + 1e-9)).sum()
    max_entropy = torch.log(torch.tensor(model.vocab_size, dtype=torch.float32))
    print(f"\nEntropy: {entropy.item():.4f} (max: {max_entropy.item():.4f})")
    print(f"Normalized entropy: {(entropy / max_entropy).item():.4f}")

if __name__ == "__main__":
    test_model_forward()