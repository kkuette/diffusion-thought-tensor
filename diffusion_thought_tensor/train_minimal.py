"""
Minimal training script for Diffusion Thought Tensor Model
Designed for quick experimentation on RTX 3090
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime

# Import our model
from model.base_model import DiffusionThoughtModel, count_parameters


class DummyTextDataset(Dataset):
    """Simple dataset for initial testing"""
    
    def __init__(self, num_samples=1000, seq_length=128, vocab_size=50257):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        # Generate random data for testing
        self.texts = torch.randint(0, vocab_size, (num_samples, seq_length))
        self.thoughts = torch.randn(num_samples, 32, 32, 16)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.texts[idx], self.thoughts[idx]


def get_noise_schedule(num_steps, schedule_type="sqrt"):
    """Generate noise schedule for diffusion"""
    if schedule_type == "linear":
        return torch.linspace(0.0001, 0.02, num_steps)
    elif schedule_type == "sqrt":
        return torch.sqrt(torch.linspace(0.0001, 0.02, num_steps))
    elif schedule_type == "cosine":
        steps = torch.arange(num_steps + 1) / num_steps
        alphas = torch.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
        alphas = alphas / alphas[0]
        betas = 1 - (alphas[1:] / alphas[:-1])
        return torch.clamp(betas, 0.0001, 0.02)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def add_noise(tokens, thought_tensor, timestep, noise_schedule):
    """Add noise to tokens and thoughts for diffusion training"""
    batch_size = tokens.shape[0]
    device = tokens.device
    
    # Get noise level for this timestep
    noise_level = noise_schedule[timestep]
    
    # For tokens: randomly replace with random tokens
    mask = torch.rand(tokens.shape, device=device) < noise_level.unsqueeze(1)
    random_tokens = torch.randint_like(tokens, 0, 50257)
    noisy_tokens = torch.where(mask, random_tokens, tokens)
    
    # For thoughts: add Gaussian noise
    thought_noise = torch.randn_like(thought_tensor) * noise_level.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    noisy_thoughts = thought_tensor + thought_noise
    
    return noisy_tokens, noisy_thoughts


def compute_loss(model, batch, noise_schedule, device):
    """Compute diffusion training loss"""
    tokens, thoughts = batch
    batch_size = tokens.shape[0]
    
    # Sample random timesteps
    timesteps = torch.randint(0, len(noise_schedule), (batch_size,), device=device)
    
    # Add noise
    noisy_tokens, noisy_thoughts = add_noise(tokens, thoughts, timesteps, noise_schedule.to(device))
    
    # Forward pass
    with autocast():
        pred_logits, pred_thoughts = model(noisy_tokens, noisy_thoughts, timesteps)
    
    # Token prediction loss
    token_loss = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.shape[-1]),
        tokens.reshape(-1)
    )
    
    # Thought reconstruction loss
    thought_loss = F.mse_loss(pred_thoughts, thoughts)
    
    # Combined loss
    total_loss = token_loss + 0.5 * thought_loss
    
    return total_loss, {
        'token_loss': token_loss.item(),
        'thought_loss': thought_loss.item(),
        'total_loss': total_loss.item()
    }


def train_epoch(model, dataloader, optimizer, scaler, noise_schedule, device):
    """Train for one epoch"""
    model.train()
    epoch_losses = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Move batch to device
        tokens, thoughts = batch
        tokens = tokens.to(device)
        thoughts = thoughts.to(device)
        
        # Compute loss
        loss, loss_dict = compute_loss(model, (tokens, thoughts), noise_schedule, device)
        
        # Backward pass with mixed precision
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Update progress bar
        epoch_losses.append(loss_dict)
        pbar.set_postfix({
            'token_loss': f"{loss_dict['token_loss']:.4f}",
            'thought_loss': f"{loss_dict['thought_loss']:.4f}"
        })
    
    return epoch_losses


def main():
    # Configuration
    config = {
        'batch_size': 4,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'num_diffusion_steps': 500,
        'noise_schedule': 'sqrt',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'checkpoint_dir': 'diffusion_thought_tensor/outputs/checkpoints',
        'log_dir': 'diffusion_thought_tensor/outputs/logs'
    }
    
    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Initialize model
    print("Initializing model...")
    model = DiffusionThoughtModel(
        vocab_size=50257,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        max_seq_length=128
    )
    model = model.to(config['device'])
    
    param_count = count_parameters(model)
    print(f"Model initialized with {param_count / 1e6:.1f}M parameters")
    
    # Create dummy dataset
    print("Creating dataset...")
    dataset = DummyTextDataset(num_samples=1000, seq_length=128)
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    # Initialize optimizer and scaler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    scaler = GradScaler()
    
    # Get noise schedule
    noise_schedule = get_noise_schedule(config['num_diffusion_steps'], config['noise_schedule'])
    
    # Training loop
    print(f"\nStarting training on {config['device']}...")
    all_losses = []
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train for one epoch
        epoch_losses = train_epoch(
            model, dataloader, optimizer, scaler, noise_schedule, config['device']
        )
        all_losses.extend(epoch_losses)
        
        # Calculate average losses
        avg_token_loss = np.mean([l['token_loss'] for l in epoch_losses])
        avg_thought_loss = np.mean([l['thought_loss'] for l in epoch_losses])
        avg_total_loss = np.mean([l['total_loss'] for l in epoch_losses])
        
        print(f"Epoch {epoch + 1} - Avg Token Loss: {avg_token_loss:.4f}, "
              f"Avg Thought Loss: {avg_thought_loss:.4f}, "
              f"Avg Total Loss: {avg_total_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': all_losses,
                'config': config
            }
            checkpoint_path = os.path.join(
                config['checkpoint_dir'],
                f'checkpoint_epoch_{epoch + 1}.pt'
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(config['checkpoint_dir'], 'final_model.pt')
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    # Save training logs
    log_path = os.path.join(config['log_dir'], f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(log_path, 'w') as f:
        json.dump({
            'config': config,
            'losses': all_losses,
            'param_count': param_count
        }, f, indent=2)
    print(f"Training logs saved to {log_path}")


if __name__ == "__main__":
    main()