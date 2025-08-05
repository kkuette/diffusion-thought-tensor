"""
Real Data Training Pipeline for Diffusion Thought Tensor Model
Training with real text data instead of synthetic patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import os
import json
import yaml
import wandb
from datetime import datetime
import time
from typing import Dict, List, Optional, Tuple

# Import our models and utilities
from model.phase2_model import Phase2DiffusionThoughtModel, count_parameters
from utils.noise_schedules import create_noise_scheduler
from data.real_data_loader import create_real_data_loaders
from train_phase2 import compute_phase2_loss, evaluate_model, save_checkpoint, load_checkpoint


def train_with_real_data(
    config_path: str = 'configs/phase2_config.yaml',
    data_source: str = 'tinystories',
    num_epochs: int = 10,
    train_samples: int = 20000,
    eval_samples: int = 2000,
    resume_checkpoint: Optional[str] = None
):
    """Train the model with real data"""
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override some settings for real data training
    config['training']['num_epochs'] = num_epochs
    config['data_source'] = data_source
    config['train_samples'] = train_samples
    config['eval_samples'] = eval_samples
    
    # Initialize wandb
    run_name = f"real-data-{data_source}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    try:
        wandb.init(
            project="diffusion-thought-tensor",
            name=run_name,
            config=config,
            tags=["real-data", data_source, "phase2"]
        )
        print("✅ Weights & Biases initialized")
    except Exception as e:
        print(f"⚠️  Wandb not available: {e}")
    
    # Setup directories
    checkpoint_dir = os.path.join('outputs', 'real_data_checkpoints')
    log_dir = os.path.join('outputs', 'real_data_logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Device setup
    device = torch.device(config['hardware']['device'])
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing Phase 2 model for real data training...")
    model = Phase2DiffusionThoughtModel(config_path=config_path)
    model = model.to(device)
    
    param_count = count_parameters(model)
    print(f"Model initialized with {param_count / 1e6:.1f}M parameters")
    
    # Create real data loaders
    print(f"Creating real data loaders (source: {data_source})...")
    train_loader, eval_loader = create_real_data_loaders(
        config, train_samples, eval_samples, data_source
    )
    
    # Initialize training components
    noise_scheduler = create_noise_scheduler(config['diffusion'])
    # Move noise scheduler to device
    if hasattr(noise_scheduler, 'alphas_cumprod'):
        noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)
    if hasattr(noise_scheduler, 'sqrt_alphas_cumprod'):
        noise_scheduler.sqrt_alphas_cumprod = noise_scheduler.sqrt_alphas_cumprod.to(device)
    if hasattr(noise_scheduler, 'sqrt_one_minus_alphas_cumprod'):
        noise_scheduler.sqrt_one_minus_alphas_cumprod = noise_scheduler.sqrt_one_minus_alphas_cumprod.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        betas=config['optimizer']['betas'],
        weight_decay=float(config['optimizer']['weight_decay']),
        eps=float(config['optimizer']['eps'])
    )
    
    scaler = GradScaler() if config['hardware']['mixed_precision'] else None
    
    # Learning rate scheduler
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(config['training']['warmup_steps']),
        T_mult=2,
        eta_min=float(config['training']['learning_rate']) * 0.01
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        start_epoch, _, _ = load_checkpoint(resume_checkpoint, model, optimizer, scaler)
        print(f"Resumed from checkpoint at epoch {start_epoch}")
    
    # Training setup
    print(f"\n🚀 Starting real data training:")
    print(f"  Model: {param_count / 1e6:.1f}M parameters")
    print(f"  Data source: {data_source}")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Evaluation samples: {len(eval_loader.dataset)}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    
    # Training loop
    all_train_losses = []
    all_eval_metrics = []
    best_eval_loss = float('inf')
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")
        
        # Training phase
        model.train()
        epoch_losses = []
        
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(train_pbar):
            # Move batch to device
            tokens, thought_stacks = batch
            tokens = tokens.to(device, non_blocking=True)
            thought_stacks = thought_stacks.to(device, non_blocking=True)
            
            # Compute loss
            loss, metrics = compute_phase2_loss(
                model, (tokens, thought_stacks), noise_scheduler, device
            )
            
            # Scale for gradient accumulation
            gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(config['training']['max_grad_norm'])
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(config['training']['max_grad_norm'])
                    )
                    optimizer.step()
                
                optimizer.zero_grad()
                lr_scheduler.step()
                
                metrics['grad_norm'] = grad_norm.item()
                metrics['learning_rate'] = lr_scheduler.get_last_lr()[0]
            
            epoch_losses.append(metrics)
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'token': f"{metrics['token_loss']:.4f}",
                'thought': f"{metrics['thought_loss']:.4f}",
                'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}",
                'mem': f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB"
            })
            
            # Log to wandb
            if wandb.run is not None:
                step = epoch * len(train_loader) + batch_idx
                wandb.log({
                    'train_step': step,
                    'epoch': epoch + 1,
                    **{f'train/{k}': v for k, v in metrics.items()}
                })
            
            # Memory cleanup
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        all_train_losses.extend(epoch_losses)
        
        # Compute epoch averages
        epoch_metrics = {}
        for key in epoch_losses[0].keys():
            epoch_metrics[f'avg_{key}'] = np.mean([l[key] for l in epoch_losses])
        
        # Evaluation phase
        print("\n📊 Evaluating model...")
        eval_metrics = evaluate_model(model, eval_loader, noise_scheduler, device)
        all_eval_metrics.append({
            'epoch': epoch + 1,
            **eval_metrics
        })
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        
        print(f"\n📈 Epoch {epoch + 1} Results:")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {epoch_metrics['avg_total_loss']:.4f}")
        print(f"    - Token Loss: {epoch_metrics['avg_token_loss']:.4f}")
        print(f"    - Thought Loss: {epoch_metrics['avg_thought_loss']:.4f}")
        print(f"    - Diversity Loss: {epoch_metrics['avg_diversity_loss']:.4f}")
        print(f"  Eval Loss: {eval_metrics['total_loss']:.4f}")
        print(f"    - Token Loss: {eval_metrics['token_loss']:.4f}")
        print(f"    - Thought Loss: {eval_metrics['thought_loss']:.4f}")
        print(f"  Learning Rate: {lr_scheduler.get_last_lr()[0]:.2e}")
        print(f"  Memory Usage: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                'epoch': epoch + 1,
                'epoch_time': epoch_time,
                **{f'train_epoch/{k}': v for k, v in epoch_metrics.items()},
                **{f'eval/{k}': v for k, v in eval_metrics.items()}
            })
        
        # Save best model
        if eval_metrics['total_loss'] < best_eval_loss:
            best_eval_loss = eval_metrics['total_loss']
            best_model_path = os.path.join(checkpoint_dir, f'best_model_{data_source}.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 New best model saved! (eval_loss: {best_eval_loss:.4f})")
        
        # Save checkpoint every epoch for real data training
        checkpoint_path = save_checkpoint(
            model, optimizer, scaler, epoch + 1,
            (epoch + 1) * len(train_loader),
            config, epoch_metrics, checkpoint_dir
        )
        print(f"💾 Checkpoint saved: {os.path.basename(checkpoint_path)}")
        
        # Memory cleanup
        torch.cuda.empty_cache()
    
    # Training complete
    total_training_time = time.time() - training_start_time
    
    print(f"\n🎉 Real data training complete!")
    print(f"  Total time: {total_training_time / 3600:.1f} hours")
    print(f"  Final eval loss: {all_eval_metrics[-1]['total_loss']:.4f}")
    print(f"  Best eval loss: {best_eval_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, f'final_model_{data_source}.pt')
    torch.save(model.state_dict(), final_model_path)
    
    # Save comprehensive training log
    log_data = {
        'config': config,
        'training_summary': {
            'data_source': data_source,
            'train_samples': train_samples,
            'eval_samples': eval_samples,
            'epochs_completed': num_epochs,
            'total_training_time_hours': total_training_time / 3600,
            'final_eval_loss': all_eval_metrics[-1]['total_loss'],
            'best_eval_loss': best_eval_loss,
            'parameter_count': param_count
        },
        'epoch_metrics': all_eval_metrics,
        'final_train_metrics': epoch_metrics
    }
    
    log_path = os.path.join(
        log_dir,
        f'real_data_training_{data_source}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"📊 Training log saved: {log_path}")
    
    if wandb.run is not None:
        # Log final summary
        wandb.summary.update({
            'final_eval_loss': all_eval_metrics[-1]['total_loss'],
            'best_eval_loss': best_eval_loss,
            'training_time_hours': total_training_time / 3600,
            'epochs_completed': num_epochs
        })
        wandb.finish()
    
    return model, all_eval_metrics, best_eval_loss


def main():
    """Main function for real data training"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Diffusion Thought Tensor with Real Data')
    parser.add_argument('--config', default='configs/phase2_config.yaml', help='Config file path')
    parser.add_argument('--data_source', default='tinystories', choices=['tinystories', 'wikitext', 'openwebtext'], help='Data source')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--train_samples', type=int, default=20000, help='Number of training samples')
    parser.add_argument('--eval_samples', type=int, default=2000, help='Number of evaluation samples')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting real data training with {args.data_source}")
    
    model, eval_metrics, best_loss = train_with_real_data(
        config_path=args.config,
        data_source=args.data_source,
        num_epochs=args.epochs,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        resume_checkpoint=args.resume
    )
    
    print(f"\n✅ Training completed successfully!")
    print(f"Best evaluation loss: {best_loss:.4f}")
    
    return model, eval_metrics


if __name__ == "__main__":
    main()