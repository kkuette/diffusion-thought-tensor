"""
Enhanced training script for Diffusion Thought Tensor Model with Thought Stacking
Supports temporal thought accumulation and memory retention
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

# Import our enhanced model
from model.enhanced_model import EnhancedDiffusionThoughtModel, count_parameters


class EnhancedTextDataset(Dataset):
    """Enhanced dataset that maintains thought stacks across sequences"""
    
    def __init__(self, num_samples=1000, seq_length=128, vocab_size=50257, 
                 thought_dims=(8, 8, 4), stack_size=8):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.thought_dims = thought_dims
        self.stack_size = stack_size
        
        # Generate random data for testing
        self.texts = torch.randint(0, vocab_size, (num_samples, seq_length))
        
        # Initialize thought stacks for each sample
        self.thought_stacks = torch.randn(num_samples, stack_size, *thought_dims)
        
        # Create some correlation between text and thought stacks
        for i in range(num_samples):
            # Use text statistics to influence thought stack
            text_mean = self.texts[i].float().mean()
            text_std = self.texts[i].float().std()
            
            # Scale thought stack based on text characteristics
            self.thought_stacks[i] *= (text_mean / vocab_size * 2 + 0.5)
            self.thought_stacks[i] += (text_std / vocab_size) * torch.randn_like(self.thought_stacks[i]) * 0.1
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.texts[idx], self.thought_stacks[idx]


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


def add_noise(tokens, thought_stack, timestep, noise_schedule):
    """Add noise to tokens and thought stacks for diffusion training"""
    batch_size = tokens.shape[0]
    device = tokens.device
    
    # Get noise level for this timestep
    timestep_cpu = timestep.cpu()
    noise_level = noise_schedule[timestep_cpu].to(device)
    
    # For tokens: randomly replace with random tokens
    vocab_size = 1000  # Match the model's vocab size
    mask = torch.rand(tokens.shape, device=device) < noise_level.unsqueeze(1)
    random_tokens = torch.randint_like(tokens, 0, vocab_size)
    noisy_tokens = torch.where(mask, random_tokens, tokens)
    
    # For thought stacks: add Gaussian noise
    stack_noise = torch.randn_like(thought_stack) * noise_level.view(-1, 1, 1, 1, 1)
    noisy_stacks = thought_stack + stack_noise
    
    return noisy_tokens, noisy_stacks


def compute_enhanced_loss(model, batch, noise_schedule, device):
    """Compute enhanced diffusion training loss with thought stack considerations"""
    tokens, thought_stacks = batch
    batch_size = tokens.shape[0]
    
    # Sample random timesteps
    timesteps = torch.randint(0, len(noise_schedule), (batch_size,), device=device)
    
    # Add noise
    noisy_tokens, noisy_stacks = add_noise(tokens, thought_stacks, timesteps, noise_schedule.to(device))
    
    # Forward pass
    with autocast():
        pred_logits, pred_thought = model(noisy_tokens, noisy_stacks, timesteps)
    
    # Token prediction loss
    token_loss = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.shape[-1]),
        tokens.reshape(-1)
    )
    
    # Thought reconstruction loss (compare with most recent thought in stack)
    target_thought = thought_stacks[:, -1]  # Latest thought in stack
    thought_loss = F.mse_loss(pred_thought, target_thought)
    
    # Memory consistency loss - encourage using information from full stack
    with torch.no_grad():
        # Get encoding from full stack
        full_encoding = model.thought_stack_encoder(thought_stacks)
        
        # Get encoding from just latest thought
        latest_only_stack = torch.zeros_like(thought_stacks)
        latest_only_stack[:, -1] = thought_stacks[:, -1]
        latest_encoding = model.thought_stack_encoder(latest_only_stack)
        
        # Memory utilization - how much the model uses historical context
        memory_utilization = 1.0 - F.cosine_similarity(full_encoding, latest_encoding, dim=-1).mean()
    
    # Thought diversity loss - encourage diverse thoughts in stack
    stack_flat = thought_stacks.view(batch_size, thought_stacks.shape[1], -1)
    thought_diversity = torch.var(stack_flat, dim=1).mean()
    diversity_loss = F.relu(0.1 - thought_diversity)  # Penalize low diversity
    
    # Combined loss
    total_loss = (token_loss + 
                  0.5 * thought_loss + 
                  0.1 * diversity_loss)
    
    return total_loss, {
        'token_loss': token_loss.item(),
        'thought_loss': thought_loss.item(),
        'diversity_loss': diversity_loss.item(),
        'memory_utilization': memory_utilization.item(),
        'total_loss': total_loss.item()
    }


def train_epoch(model, dataloader, optimizer, scaler, noise_schedule, device, epoch):
    """Train for one epoch with enhanced thought stacking"""
    model.train()
    epoch_losses = []
    
    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        tokens, thought_stacks = batch
        tokens = tokens.to(device)
        thought_stacks = thought_stacks.to(device)
        
        # Compute loss
        loss, loss_dict = compute_enhanced_loss(model, (tokens, thought_stacks), noise_schedule, device)
        
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
            'thought_loss': f"{loss_dict['thought_loss']:.4f}",
            'memory_use': f"{loss_dict['memory_utilization']:.3f}"
        })
    
    return epoch_losses


def evaluate_model(model, dataloader, noise_schedule, device):
    """Evaluate model performance"""
    model.eval()
    total_losses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            tokens, thought_stacks = batch
            tokens = tokens.to(device)
            thought_stacks = thought_stacks.to(device)
            
            loss, loss_dict = compute_enhanced_loss(model, (tokens, thought_stacks), noise_schedule, device)
            total_losses.append(loss_dict)
    
    # Average metrics
    avg_metrics = {}
    for key in total_losses[0].keys():
        avg_metrics[key] = np.mean([l[key] for l in total_losses])
    
    return avg_metrics


def main():
    """Main training function"""
    # Configuration
    config = {
        'batch_size': 4,
        'num_epochs': 20,
        'learning_rate': 1e-4,
        'num_diffusion_steps': 500,
        'noise_schedule': 'sqrt',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'checkpoint_dir': 'outputs/checkpoints',
        'log_dir': 'outputs/logs',
        
        # Enhanced model config
        'thought_stack_size': 8,
        'thought_dims': (8, 8, 4),
        'vocab_size': 1000,  # Smaller vocab for testing
        'embed_dim': 256,
        'num_layers': 6,
        'num_heads': 8,
        'max_seq_length': 64
    }
    
    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Initialize enhanced model
    print("Initializing enhanced model with thought stacking...")
    model = EnhancedDiffusionThoughtModel(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        max_seq_length=config['max_seq_length'],
        thought_dims=config['thought_dims'],
        thought_stack_size=config['thought_stack_size']
    )
    model = model.to(config['device'])
    
    param_count = count_parameters(model)
    print(f"Enhanced model initialized with {param_count / 1e6:.1f}M parameters")
    print(f"Thought stack size: {config['thought_stack_size']}")
    print(f"Thought dimensions: {config['thought_dims']}")
    
    # Create enhanced dataset
    print("Creating enhanced dataset with thought stacks...")
    train_dataset = EnhancedTextDataset(
        num_samples=1000, 
        seq_length=config['max_seq_length'],
        vocab_size=config['vocab_size'],
        thought_dims=config['thought_dims'],
        stack_size=config['thought_stack_size']
    )
    
    eval_dataset = EnhancedTextDataset(
        num_samples=200, 
        seq_length=config['max_seq_length'],
        vocab_size=config['vocab_size'],
        thought_dims=config['thought_dims'],
        stack_size=config['thought_stack_size']
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
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
    print(f"\nStarting enhanced training on {config['device']}...")
    all_train_losses = []
    all_eval_metrics = []
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train for one epoch
        epoch_losses = train_epoch(
            model, train_dataloader, optimizer, scaler, noise_schedule, config['device'], epoch + 1
        )
        all_train_losses.extend(epoch_losses)
        
        # Evaluate model
        if (epoch + 1) % 5 == 0:
            print("Evaluating model...")
            eval_metrics = evaluate_model(model, eval_dataloader, noise_schedule, config['device'])
            all_eval_metrics.append({
                'epoch': epoch + 1,
                **eval_metrics
            })
            
            print(f"Eval - Token Loss: {eval_metrics['token_loss']:.4f}, "
                  f"Thought Loss: {eval_metrics['thought_loss']:.4f}, "
                  f"Memory Utilization: {eval_metrics['memory_utilization']:.3f}")
        
        # Calculate average losses for this epoch
        avg_token_loss = np.mean([l['token_loss'] for l in epoch_losses])
        avg_thought_loss = np.mean([l['thought_loss'] for l in epoch_losses])
        avg_memory_use = np.mean([l['memory_utilization'] for l in epoch_losses])
        
        print(f"Epoch {epoch + 1} - Avg Token Loss: {avg_token_loss:.4f}, "
              f"Avg Thought Loss: {avg_thought_loss:.4f}, "
              f"Avg Memory Use: {avg_memory_use:.3f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': all_train_losses,
                'eval_metrics': all_eval_metrics,
                'config': config
            }
            checkpoint_path = os.path.join(
                config['checkpoint_dir'],
                f'enhanced_checkpoint_epoch_{epoch + 1}.pt'
            )
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(config['checkpoint_dir'], 'enhanced_final_model.pt')
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    # Save training logs
    log_path = os.path.join(config['log_dir'], f'enhanced_training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(log_path, 'w') as f:
        json.dump({
            'config': config,
            'train_losses': all_train_losses,
            'eval_metrics': all_eval_metrics,
            'param_count': param_count
        }, f, indent=2)
    print(f"Training logs saved to {log_path}")


if __name__ == "__main__":
    main()