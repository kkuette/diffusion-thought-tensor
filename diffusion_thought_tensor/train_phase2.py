"""
Phase 2 Training Pipeline for 500M Parameter Diffusion Thought Tensor Model
Optimized for RTX 3090 with advanced memory management and training techniques
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
import yaml
import wandb
from datetime import datetime
import math
import time
from typing import Dict, List, Optional, Tuple

# Import our models and utilities
from model.phase2_model import Phase2DiffusionThoughtModel, count_parameters
from utils.noise_schedules import NoiseScheduler, create_noise_scheduler


class Phase2Dataset(Dataset):
    """Advanced dataset for Phase 2 training with realistic text patterns"""
    
    def __init__(
        self, 
        num_samples: int = 10000,
        seq_length: int = 512,
        vocab_size: int = 50257,
        thought_dims: Tuple[int, ...] = (32, 32, 16),
        stack_size: int = 16,
        use_real_patterns: bool = True
    ):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.thought_dims = thought_dims
        self.stack_size = stack_size
        
        print(f"Generating {num_samples} samples for Phase 2 training...")
        
        if use_real_patterns:
            # Generate more realistic text patterns
            self.texts = self._generate_realistic_patterns()
        else:
            # Simple random generation
            self.texts = torch.randint(0, vocab_size, (num_samples, seq_length))
        
        # Generate thought stacks with better initialization
        self.thought_stacks = self._generate_thought_stacks()
        
        print(f"Dataset created with {len(self.texts)} samples")
    
    def _generate_realistic_patterns(self) -> torch.Tensor:
        """Generate more realistic text patterns"""
        texts = []
        
        # Common token frequencies (simplified approximation)
        common_tokens = list(range(50, 1000))  # More common tokens
        rare_tokens = list(range(1000, self.vocab_size))  # Rare tokens
        
        for i in tqdm(range(self.num_samples), desc="Generating realistic patterns"):
            # Start with common tokens (80% probability)
            text = []
            for _ in range(self.seq_length):
                if np.random.random() < 0.8:
                    token = np.random.choice(common_tokens)
                else:
                    token = np.random.choice(rare_tokens)
                text.append(token)
            
            # Add some structure (repeated patterns, etc.)
            if i % 100 == 0:  # Every 100th sample has repeated patterns
                pattern_length = np.random.randint(3, 10)
                pattern = text[:pattern_length]
                for j in range(0, len(text), pattern_length):
                    end_idx = min(j + pattern_length, len(text))
                    text[j:end_idx] = pattern[:end_idx-j]
            
            texts.append(text)
        
        return torch.tensor(texts, dtype=torch.long)
    
    def _generate_thought_stacks(self) -> torch.Tensor:
        """Generate thought stacks with better initialization"""
        # Initialize with Xavier/Glorot initialization
        fan_in = math.prod(self.thought_dims)
        std = math.sqrt(2.0 / fan_in)
        
        stacks = torch.randn(
            self.num_samples, 
            self.stack_size, 
            *self.thought_dims
        ) * std
        
        # Add some correlation between text and thoughts
        for i in range(self.num_samples):
            # Use text statistics to influence thought characteristics
            text_mean = self.texts[i].float().mean()
            text_var = self.texts[i].float().var()
            
            # Scale thoughts based on text characteristics
            scale_factor = (text_mean / self.vocab_size) * 2 + 0.5
            stacks[i] *= scale_factor
            
            # Add variance-based noise
            noise_scale = torch.sqrt(text_var / self.vocab_size) * 0.1
            stacks[i] += torch.randn_like(stacks[i]) * noise_scale
        
        return stacks
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.texts[idx], self.thought_stacks[idx]


def compute_phase2_loss(
    model: Phase2DiffusionThoughtModel,
    batch: Tuple[torch.Tensor, torch.Tensor],
    noise_scheduler: NoiseScheduler,
    device: torch.device,
    timestep_weights: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute advanced Phase 2 training loss with multiple objectives"""
    
    tokens, thought_stacks = batch
    batch_size = tokens.shape[0]
    
    # Sample random timesteps with optional weighting
    if timestep_weights is not None:
        # Weighted sampling - focus on difficult timesteps
        timesteps = torch.multinomial(
            timestep_weights, 
            batch_size, 
            replacement=True
        ).to(device)
    else:
        timesteps = torch.randint(
            0, noise_scheduler.num_steps, 
            (batch_size,), 
            device=device
        )
    
    # Generate noise for tokens and thoughts
    token_noise = torch.randn_like(tokens, dtype=torch.float32)
    thought_noise = torch.randn_like(thought_stacks)
    
    # Add noise according to schedule
    noisy_tokens_float = noise_scheduler.add_noise(
        tokens.float(), token_noise, timesteps
    )
    # Convert back to long for embedding lookup
    noisy_tokens = torch.clamp(noisy_tokens_float.round(), 0, model.vocab_size - 1).long()
    
    noisy_stacks = noise_scheduler.add_noise(
        thought_stacks, thought_noise, timesteps
    )
    
    # Forward pass with autocast for memory efficiency
    with autocast():
        pred_logits, pred_thought = model(noisy_tokens, noisy_stacks, timesteps)
    
    # Token denoising loss
    if noise_scheduler.prediction_type == "epsilon":
        # Predict noise
        token_target = token_noise
        thought_target = thought_noise
    elif noise_scheduler.prediction_type == "v_prediction":
        # Predict velocity
        token_target = noise_scheduler.get_velocity(tokens.float(), token_noise, timesteps)
        thought_target = noise_scheduler.get_velocity(thought_stacks, thought_noise, timesteps)
    else:  # "sample"
        # Predict original sample
        token_target = tokens.float()
        thought_target = thought_stacks
    
    # Token reconstruction loss (convert logits to appropriate target)
    if noise_scheduler.prediction_type == "epsilon":
        # For epsilon prediction, compare against noise in logit space
        pred_token_noise = pred_logits - tokens.float().unsqueeze(-1)
        token_loss = F.mse_loss(
            pred_token_noise.mean(dim=-1), 
            token_target
        )
    else:
        # For other predictions, use cross-entropy with target tokens
        token_loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.shape[-1]),
            tokens.reshape(-1)
        ) * 0.1  # Scale down for balance
    
    # Thought reconstruction loss
    if noise_scheduler.prediction_type == "epsilon":
        thought_target = thought_noise
    else:
        thought_target = thought_stacks[:, -1]  # Target latest thought
    
    thought_loss = F.mse_loss(pred_thought, thought_target)
    
    # Advanced losses for Phase 2
    
    # 1. Thought diversity loss - encourage diverse thoughts in stack
    stack_flat = thought_stacks.view(batch_size, thought_stacks.shape[1], -1)
    thought_diversity = torch.var(stack_flat, dim=1).mean()
    diversity_loss = F.relu(0.05 - thought_diversity)  # Encourage diversity > 0.05
    
    # 2. Temporal consistency loss - nearby thoughts should be similar
    temporal_diffs = torch.diff(stack_flat, dim=1)
    temporal_consistency = torch.mean(torch.norm(temporal_diffs, dim=-1))
    consistency_loss = F.relu(temporal_consistency - 1.0)  # Limit change rate
    
    # 3. Cross-modal alignment loss - thoughts should relate to text
    with torch.no_grad():
        # Get text embeddings
        text_embeds = model.token_embeddings(tokens).mean(dim=1)
        thought_embeds = model.thought_stack_encoder(thought_stacks)
        
        # Encourage alignment
        alignment = F.cosine_similarity(text_embeds, thought_embeds, dim=-1).mean()
        alignment_loss = F.relu(0.1 - alignment)  # Encourage alignment > 0.1
    
    # 4. Timestep-weighted loss - focus on difficult timesteps
    timestep_weights_tensor = torch.ones_like(timesteps, dtype=torch.float32)
    if timesteps.max() > 0:
        # Weight middle timesteps more heavily (they're often hardest)
        mid_point = noise_scheduler.num_steps // 2
        timestep_weights_tensor = 1.0 + 0.5 * torch.exp(
            -((timesteps.float() - mid_point) / (mid_point * 0.3)) ** 2
        )
    
    # Combine losses with careful weighting
    total_loss = (
        token_loss * timestep_weights_tensor.mean() +
        thought_loss * 0.5 +
        diversity_loss * 0.1 +
        consistency_loss * 0.05 +
        alignment_loss * 0.1
    )
    
    # Collect metrics
    metrics = {
        'token_loss': token_loss.item(),
        'thought_loss': thought_loss.item(),
        'diversity_loss': diversity_loss.item(),
        'consistency_loss': consistency_loss.item(),
        'alignment_loss': alignment_loss.item(),
        'total_loss': total_loss.item(),
        'thought_diversity': thought_diversity.item(),
        'temporal_consistency': temporal_consistency.item(),
        'alignment_score': alignment.item() if 'alignment' in locals() else 0.0
    }
    
    return total_loss, metrics


def train_epoch(
    model: Phase2DiffusionThoughtModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    noise_scheduler: NoiseScheduler,
    device: torch.device,
    epoch: int,
    config: Dict,
    timestep_weights: Optional[torch.Tensor] = None
) -> List[Dict[str, float]]:
    """Train for one epoch with advanced techniques"""
    
    model.train()
    epoch_losses = []
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    max_grad_norm = config['training']['max_grad_norm']
    
    # Progress tracking
    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    total_tokens_processed = 0
    
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        tokens, thought_stacks = batch
        tokens = tokens.to(device, non_blocking=True)
        thought_stacks = thought_stacks.to(device, non_blocking=True)
        
        # Compute loss
        loss, metrics = compute_phase2_loss(
            model, (tokens, thought_stacks), noise_scheduler, device, timestep_weights
        )
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward pass with mixed precision
        scaler.scale(loss).backward()
        
        # Accumulate gradients
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update metrics
            metrics['grad_norm'] = grad_norm.item()
        
        # Memory management
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
        
        # Update progress
        total_tokens_processed += tokens.numel()
        epoch_losses.append(metrics)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['total_loss']:.4f}",
            'token_loss': f"{metrics['token_loss']:.4f}",
            'thought_loss': f"{metrics['thought_loss']:.4f}",
            'mem_gb': f"{torch.cuda.memory_allocated() / 1024**3:.1f}"
        })
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                'train_step': epoch * len(dataloader) + batch_idx,
                **{f'train/{k}': v for k, v in metrics.items()}
            })
    
    return epoch_losses


def evaluate_model(
    model: Phase2DiffusionThoughtModel,
    dataloader: DataLoader,
    noise_scheduler: NoiseScheduler,
    device: torch.device
) -> Dict[str, float]:
    """Comprehensive model evaluation"""
    
    model.eval()
    total_losses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            tokens, thought_stacks = batch
            tokens = tokens.to(device, non_blocking=True)
            thought_stacks = thought_stacks.to(device, non_blocking=True)
            
            loss, metrics = compute_phase2_loss(
                model, (tokens, thought_stacks), noise_scheduler, device
            )
            total_losses.append(metrics)
    
    # Average all metrics
    avg_metrics = {}
    for key in total_losses[0].keys():
        avg_metrics[key] = np.mean([l[key] for l in total_losses])
    
    return avg_metrics


def save_checkpoint(
    model: Phase2DiffusionThoughtModel,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    epoch: int,
    step: int,
    config: Dict,
    metrics: Dict,
    checkpoint_dir: str
) -> str:
    """Save model checkpoint with all necessary information"""
    
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f'phase2_checkpoint_epoch_{epoch}_step_{step}.pt'
    )
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'config': config,
        'metrics': metrics,
        'model_config': model.config,
        'param_count': count_parameters(model),
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: Phase2DiffusionThoughtModel,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler
) -> Tuple[int, int, Dict]:
    """Load checkpoint and resume training"""
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    epoch = checkpoint['epoch']
    step = checkpoint.get('step', 0)
    metrics = checkpoint.get('metrics', {})
    
    print(f"Resumed from epoch {epoch}, step {step}")
    return epoch, step, metrics


def main():
    """Main Phase 2 training function"""
    
    # Load configuration
    config_path = 'configs/phase2_config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize wandb if available
    try:
        wandb.init(
            project="diffusion-thought-tensor",
            name=f"phase2-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=config
        )
        print("✅ Weights & Biases initialized")
    except Exception as e:
        print(f"⚠️  Wandb not available: {e}")
    
    # Setup directories
    checkpoint_dir = config['training'].get('checkpoint_dir', 'outputs/phase2_checkpoints')
    log_dir = config['training'].get('log_dir', 'outputs/phase2_logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Device setup
    device = torch.device(config['hardware']['device'])
    print(f"Using device: {device}")
    
    # Initialize Phase 2 model
    print("Initializing Phase 2 model...")
    model = Phase2DiffusionThoughtModel(config_path=config_path)
    model = model.to(device)
    
    param_count = count_parameters(model)
    print(f"Phase 2 model initialized with {param_count / 1e6:.1f}M parameters")
    
    # Setup data
    print("Creating Phase 2 datasets...")
    train_dataset = Phase2Dataset(
        num_samples=20000,  # Larger dataset for Phase 2
        seq_length=config['model']['max_seq_length'],
        vocab_size=config['model']['vocab_size'],
        thought_dims=tuple(config['thought_tensor']['input_dims']),
        stack_size=config['thought_tensor']['stack_size']
    )
    
    eval_dataset = Phase2Dataset(
        num_samples=2000,
        seq_length=config['model']['max_seq_length'],
        vocab_size=config['model']['vocab_size'],
        thought_dims=tuple(config['thought_tensor']['input_dims']),
        stack_size=config['thought_tensor']['stack_size']
    )
    
    # Data loaders with optimized settings
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware'].get('dataloader_pin_memory', True),
        persistent_workers=True if config['hardware']['num_workers'] > 0 else False
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware'].get('dataloader_pin_memory', True),
        persistent_workers=True if config['hardware']['num_workers'] > 0 else False
    )
    
    # Initialize noise scheduler
    noise_scheduler = create_noise_scheduler(config['diffusion'])
    print(f"Noise scheduler: {config['diffusion']['schedule_type']} with {config['diffusion']['num_steps']} steps")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=config['optimizer']['betas'],
        weight_decay=config['optimizer']['weight_decay'],
        eps=config['optimizer']['eps']
    )
    
    # Initialize mixed precision scaler
    scaler = GradScaler() if config['hardware']['mixed_precision'] else None
    
    # Learning rate scheduler
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=config['training']['warmup_steps'],
        T_mult=2,
        eta_min=config['training']['learning_rate'] * 0.01
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    resume_checkpoint = config.get('resume_checkpoint')
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        start_epoch, _, _ = load_checkpoint(resume_checkpoint, model, optimizer, scaler)
    
    # Training loop
    print(f"\\nStarting Phase 2 training from epoch {start_epoch}...")
    print(f"Model size: {param_count / 1e6:.1f}M parameters")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Evaluation samples: {len(eval_dataset)}")
    
    num_epochs = config['training'].get('num_epochs', 100)
    eval_every = config['training']['eval_every']
    save_every = config['training']['save_every']
    
    all_train_losses = []
    all_eval_metrics = []
    best_eval_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        print(f"\\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train for one epoch
        epoch_losses = train_epoch(
            model, train_dataloader, optimizer, scaler, 
            noise_scheduler, device, epoch + 1, config
        )
        all_train_losses.extend(epoch_losses)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluation
        if (epoch + 1) % eval_every == 0:
            print("Evaluating model...")
            eval_metrics = evaluate_model(model, eval_dataloader, noise_scheduler, device)
            all_eval_metrics.append({
                'epoch': epoch + 1,
                **eval_metrics
            })
            
            print(f"Eval - Total Loss: {eval_metrics['total_loss']:.4f}, "
                  f"Token Loss: {eval_metrics['token_loss']:.4f}, "
                  f"Thought Loss: {eval_metrics['thought_loss']:.4f}")
            
            # Log to wandb
            if wandb.run is not None:
                wandb.log({
                    'epoch': epoch + 1,
                    **{f'eval/{k}': v for k, v in eval_metrics.items()}
                })
            
            # Save best model
            if eval_metrics['total_loss'] < best_eval_loss:
                best_eval_loss = eval_metrics['total_loss']
                best_model_path = os.path.join(checkpoint_dir, 'phase2_best_model.pt')
                torch.save(model.state_dict(), best_model_path)
                print(f"💾 New best model saved to {best_model_path}")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            avg_metrics = {}
            for key in epoch_losses[0].keys():
                avg_metrics[f'avg_{key}'] = np.mean([l[key] for l in epoch_losses])
            
            checkpoint_path = save_checkpoint(
                model, optimizer, scaler, epoch + 1, 
                (epoch + 1) * len(train_dataloader),
                config, avg_metrics, checkpoint_dir
            )
            print(f"💾 Checkpoint saved to {checkpoint_path}")
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_token_loss = np.mean([l['token_loss'] for l in epoch_losses])
        avg_thought_loss = np.mean([l['thought_loss'] for l in epoch_losses])
        avg_total_loss = np.mean([l['total_loss'] for l in epoch_losses])
        
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Avg Total Loss: {avg_total_loss:.4f}")
        print(f"  Avg Token Loss: {avg_token_loss:.4f}")
        print(f"  Avg Thought Loss: {avg_thought_loss:.4f}")
        print(f"  Learning Rate: {lr_scheduler.get_last_lr()[0]:.2e}")
        print(f"  Memory: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, 'phase2_final_model.pt')
    torch.save(model.state_dict(), final_path)
    print(f"\\n🎉 Phase 2 training complete! Final model saved to {final_path}")
    
    # Save training logs
    log_path = os.path.join(
        log_dir, 
        f'phase2_training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    with open(log_path, 'w') as f:
        json.dump({
            'config': config,
            'param_count': param_count,
            'training_time_epochs': num_epochs - start_epoch,
            'final_metrics': all_eval_metrics[-1] if all_eval_metrics else {},
            'best_eval_loss': best_eval_loss
        }, f, indent=2)
    
    print(f"📊 Training logs saved to {log_path}")
    
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()