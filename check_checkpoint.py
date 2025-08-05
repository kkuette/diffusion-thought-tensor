#!/usr/bin/env python3
"""
Check checkpoint contents
"""

import torch
import os

checkpoint_path = "./diffusion_thought_tensor/outputs/masked_checkpoints/best_masked_model.pt"

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    print("\nCheckpoint keys:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    if "epoch" in checkpoint:
        print(f"\nEpoch: {checkpoint['epoch']}")
    
    if "loss" in checkpoint:
        print(f"Loss: {checkpoint['loss']}")
        
    if "best_loss" in checkpoint:
        print(f"Best loss: {checkpoint['best_loss']}")
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"\nModel state dict has {len(state_dict)} keys")
        
        # Check first few layer statistics
        for i, (name, param) in enumerate(state_dict.items()):
            if i < 5:
                print(f"  {name}: shape={param.shape}, mean={param.mean().item():.4f}, std={param.std().item():.4f}")
else:
    print(f"Checkpoint not found at {checkpoint_path}")