"""
Training utilities for DREAM-Enhanced Model Training
"""

import torch
import torch.nn as nn
import os
from typing import Optional


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in the checkpoint directory"""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.endswith(".pt") and "dream_checkpoint_epoch_" in f
    ]
    if not checkpoint_files:
        return None

    # Extract epoch numbers and find the latest
    epochs = []
    for f in checkpoint_files:
        try:
            epoch_num = int(f.split("_epoch_")[1].split(".pt")[0])
            epochs.append((epoch_num, f))
        except (ValueError, IndexError):
            continue

    if not epochs:
        return None

    _, latest_file = max(epochs)
    return os.path.join(checkpoint_dir, latest_file)


def load_checkpoint(checkpoint_path: str, model, optimizer, lr_scheduler, device):
    """Load checkpoint and return starting epoch and metrics"""
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if "scheduler_state_dict" in checkpoint:
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint.get("epoch", 0)
    best_metrics = checkpoint.get("best_metrics", {})
    best_eval_loss = best_metrics.get("total_loss", float("inf"))

    print(f"✅ Checkpoint loaded successfully!")
    print(f"   Resuming from epoch: {start_epoch}")
    print(f"   Best eval loss so far: {best_eval_loss:.4f}")

    return start_epoch, best_eval_loss, checkpoint.get("metrics", {})