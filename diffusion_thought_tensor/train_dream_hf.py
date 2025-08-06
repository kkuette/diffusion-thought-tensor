"""
DREAM-Enhanced Model Training with Hugging Face Datasets
Train the DREAM-Enhanced Diffusion Thought Tensor Model with real data
Includes bidirectional attention, self-learning thoughts, and adaptive features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import os
import json
import yaml
import wandb
from datetime import datetime
import time
import math
from typing import Dict, List, Optional, Tuple

# Import DREAM-enhanced models and utilities
from model.stacked_3d_model import StackedDiffusionModel3D, MaskingStrategy
from utils.noise_schedules import create_noise_scheduler
from utils.thought_analyzer import create_thought_analyzer
from utils.training_utils import count_parameters, find_latest_checkpoint, load_checkpoint
from utils.logging_utils import DreamTrainingLogger
from utils.loss_utils import compute_dream_enhanced_loss
from utils.memory_utils import MemoryMonitor
from data.huggingface_data_loader import (
    create_huggingface_data_loaders,
    SUPPORTED_DATASETS,
)




def main():
    """Main function for DREAM-enhanced model training"""

    import argparse

    parser = argparse.ArgumentParser(description="Train DREAM-Enhanced Model")
    parser.add_argument(
        "--config", default="configs/enhanced_config.yaml", help="Config file path"
    )
    parser.add_argument(
        "--dataset", default="roneneldan/TinyStories", help="Dataset name"
    )
    parser.add_argument(
        "--train_samples", type=int, default=5000, help="Number of training samples"
    )
    parser.add_argument(
        "--eval_samples", type=int, default=500, help="Number of evaluation samples"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--auto_resume", action="store_true", help="Automatically resume from latest checkpoint"
    )
    parser.add_argument(
        "--mask_ratio", type=float, default=0.7, help="Token masking ratio"
    )
    parser.add_argument(
        "--confidence_threshold", type=float, default=0.75, help="Confidence threshold for unmasking"
    )

    args = parser.parse_args()

    print(f"🎭 Starting DREAM-enhanced model training with {args.dataset}")
    print(f"   Mask ratio: {args.mask_ratio}")
    print(f"   Confidence threshold: {args.confidence_threshold}")

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["training"]["num_epochs"] = args.epochs

    # Setup directories
    checkpoint_dir = config["checkpointing"]["checkpoint_dir"].replace("enhanced_", "dream_")
    log_dir = config["checkpointing"]["log_dir"].replace("enhanced_", "dream_")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Device setup
    device = torch.device(config["hardware"]["device"])
    print(f"Using device: {device}")

    # Get pad_token_id
    pad_token_id = 50256  # GPT2's eos_token_id used as pad_token

    # Create DREAM masking strategy
    masking_strategy = MaskingStrategy(
        mask_ratio=args.mask_ratio,
        confidence_threshold=args.confidence_threshold,
        progressive_unmasking=True,
        block_size=4,
    )

    # Initialize DREAM-enhanced 3D model
    print("Initializing DREAM-Enhanced 3D Model...")
    
    # Map 3D config to 2D spatial dimensions (H, W) and separate depth
    original_dims = config["thought_tensor"]["input_dims"]  # e.g., [32, 32, 16]
    thought_dims = (original_dims[0], original_dims[1])  # (H, W)
    stack_depth = config["thought_tensor"]["stack_size"]
    thought_hidden_dim = original_dims[2] if len(original_dims) > 2 else 256
    
    model = StackedDiffusionModel3D(
        vocab_size=config["model"].get("vocab_size", 50257),
        embed_dim=config["model"]["embed_dim"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        thought_dims=thought_dims,  # (H, W) for each 2D thought
        stack_depth=stack_depth,  # Number of thoughts in the 3D stack
        thought_hidden_dim=thought_hidden_dim,  # Channel dimension for thoughts
        max_seq_length=config["model"]["max_seq_length"],
        dropout=config["model"]["dropout"],
        masking_strategy=masking_strategy,
    ).to(device).float()

    param_count = count_parameters(model)
    print(f"DREAM model initialized with {param_count / 1e6:.1f}M parameters")

    # Create data loaders
    try:
        print(f"Creating Hugging Face data loaders...")
        train_loader, eval_loader = create_huggingface_data_loaders(
            config,
            dataset_name=args.dataset,
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
            tokenizer_name="gpt2",
        )

        # Update vocab size if needed
        actual_vocab_size = train_loader.dataset.vocab_size
        if actual_vocab_size != model.vocab_size:
            print(f"Updating model vocab size: {model.vocab_size} -> {actual_vocab_size}")
            old_embeddings = model.token_embed
            model.token_embed = nn.Embedding(actual_vocab_size, model.embed_dim).to(device)
            min_vocab = min(old_embeddings.num_embeddings, actual_vocab_size)
            model.token_embed.weight.data[:min_vocab] = old_embeddings.weight.data[:min_vocab]

            model.output_proj = nn.Linear(model.embed_dim, actual_vocab_size).to(device)
            model.vocab_size = actual_vocab_size

    except Exception as e:
        print(f"Error loading Hugging Face dataset: {e}")
        return

    # Initialize training components
    noise_scheduler = create_noise_scheduler(config["diffusion"])
    if hasattr(noise_scheduler, "sqrt_alphas_cumprod"):
        noise_scheduler.sqrt_alphas_cumprod = noise_scheduler.sqrt_alphas_cumprod.to(device)
        noise_scheduler.sqrt_one_minus_alphas_cumprod = noise_scheduler.sqrt_one_minus_alphas_cumprod.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        betas=config["optimizer"]["betas"],
        weight_decay=float(config["optimizer"]["weight_decay"]),
        eps=float(config["optimizer"]["eps"]),
    )

    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
    
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(config["training"]["warmup_steps"]),
        T_mult=2,
        eta_min=float(config["training"]["learning_rate"]) * 0.01,
    )
    
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
        min_lr=float(config["training"]["learning_rate"]) * 0.001
    )
    
    lr_scheduler = cosine_scheduler

    # Checkpoint loading
    start_epoch = 0
    best_eval_loss = float("inf")
    resume_checkpoint = None

    if args.resume:
        resume_checkpoint = args.resume
    elif args.auto_resume:
        resume_checkpoint = find_latest_checkpoint(checkpoint_dir)

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        try:
            start_epoch, best_eval_loss, _ = load_checkpoint(
                resume_checkpoint, model, optimizer, lr_scheduler, device
            )
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            start_epoch = 0
            best_eval_loss = float("inf")

    # Training configuration
    num_epochs = config["training"]["num_epochs"]
    gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
    max_grad_norm = float(config["training"]["max_grad_norm"])
    early_stopping_patience = config["training"].get("early_stopping_patience", 3)
    
    # Progressive masking and confidence scheduling
    initial_mask_ratio = args.mask_ratio
    final_mask_ratio = args.mask_ratio * 0.5
    initial_confidence = args.confidence_threshold
    final_confidence = min(0.9, args.confidence_threshold + 0.15)

    # Initialize mixed precision and monitoring
    scaler = GradScaler("cuda") if config.get("hardware", {}).get("mixed_precision", True) else None
    use_amp = scaler is not None
    memory_monitor = MemoryMonitor(target_memory_gb=20)
    
    # Initialize DREAM logger
    use_wandb = config.get("logging", {}).get("use_wandb", False)
    use_tensorboard = config.get("logging", {}).get("use_tensorboard", True)
    logger = DreamTrainingLogger(log_dir, config, use_wandb, use_tensorboard)

    # Training info
    print(f"\n🎭 Starting DREAM-enhanced training:")
    print(f"  Model: {param_count / 1e6:.1f}M parameters")
    print(f"  Dataset: {args.dataset} ({len(train_loader.dataset)} train, {len(eval_loader.dataset)} eval)")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Masking: {initial_mask_ratio:.2f} → {final_mask_ratio:.2f}")
    print(f"  Confidence: {initial_confidence:.2f} → {final_confidence:.2f}")
    print(f"  Mixed Precision: {'Enabled' if use_amp else 'Disabled'}")

    training_start_time = time.time()
    no_improvement_count = 0
    stack_reset_frequency = config.get("training", {}).get("stack_reset_frequency", 10)
    persistent_thought_stacks = None
    batch_count = 0

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        # Calculate progressive parameters
        progress = epoch / max(1, num_epochs - 1)
        current_mask_ratio = initial_mask_ratio + (final_mask_ratio - initial_mask_ratio) * progress
        current_confidence = initial_confidence + (final_confidence - initial_confidence) * progress
        
        print(f"Using mask ratio: {current_mask_ratio:.3f}, confidence: {current_confidence:.3f}")

        # Training phase
        model.train()
        epoch_losses = []

        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(train_pbar):
            tokens = batch.to(device, non_blocking=True)
            batch_size = tokens.shape[0]

            # Handle persistent thought stacks
            if (
                batch_count % stack_reset_frequency == 0
                or persistent_thought_stacks is None
                or persistent_thought_stacks.shape[0] != batch_size
            ):
                current_thought_stacks = model.initialize_stack(batch_size, device)
            else:
                current_thought_stacks = persistent_thought_stacks

            # Compute loss with mixed precision
            if use_amp:
                with autocast("cuda"):
                    loss, metrics, evolved_stacks = compute_dream_enhanced_loss(
                        model, tokens, current_thought_stacks, noise_scheduler,
                        device, epoch, config, current_mask_ratio, current_confidence, pad_token_id
                    )
                    loss = loss / gradient_accumulation_steps

                scaler.scale(loss).backward()
            else:
                loss, metrics, evolved_stacks = compute_dream_enhanced_loss(
                    model, tokens, current_thought_stacks, noise_scheduler,
                    device, epoch, config, current_mask_ratio, current_confidence, pad_token_id
                )
                loss = loss / gradient_accumulation_steps
                loss.backward()

            persistent_thought_stacks = evolved_stacks.detach()
            batch_count += 1

            # Optimizer step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

                metrics["grad_norm"] = grad_norm.item()
                metrics["learning_rate"] = lr_scheduler.get_last_lr()[0]

            epoch_losses.append(metrics)

            # Log metrics
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 50 == 0:
                logger.log_metrics(metrics, global_step, "train", model)

            # Update progress bar
            progress_info = {
                "loss": f"{metrics['total_loss']:.4f}",
                "tok_loss": f"{metrics['token_loss']:.4f}",
                "m_acc": f"{metrics['masked_accuracy']:.3f}",
                "u_acc": f"{metrics['unmasked_accuracy']:.3f}",
                "t_rms": f"{metrics['thought_rms']:.3f}",
                "mask": f"{metrics['mask_ratio_actual']:.2f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
            }
            
            # Add DREAM-specific metrics to progress
            if "bidirectional_effectiveness" in metrics:
                progress_info["bi_eff"] = f"{metrics['bidirectional_effectiveness']:.2f}"
            if "confidence_threshold_effectiveness" in metrics:
                progress_info["conf_eff"] = f"{metrics['confidence_threshold_effectiveness']:.2f}"
                
            train_pbar.set_postfix(progress_info)

            # Memory cleanup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        # Compute epoch averages
        epoch_metrics = {}
        for key in epoch_losses[0].keys():
            values = []
            for l in epoch_losses:
                if key not in l:
                    continue  # Skip missing keys
                val = l[key]
                if torch.is_tensor(val):
                    val = float(val.detach().cpu().item()) if val.numel() == 1 else val.detach().cpu().numpy()
                elif hasattr(val, 'item'):
                    val = float(val.item())
                values.append(val)
            if values:  # Only compute mean if we have values
                epoch_metrics[f"avg_{key}"] = np.mean(values)

        # Evaluation phase
        print("\n📊 Evaluating DREAM model...")
        model.eval()
        eval_losses = []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                tokens = batch.to(device, non_blocking=True)
                batch_size = tokens.shape[0]
                eval_thought_stacks = model.initialize_stack(batch_size, device)

                if use_amp:
                    with autocast("cuda"):
                        loss, metrics, _ = compute_dream_enhanced_loss(
                            model, tokens, eval_thought_stacks, noise_scheduler,
                            device, epoch, config, args.mask_ratio, args.confidence_threshold, pad_token_id
                        )
                else:
                    loss, metrics, _ = compute_dream_enhanced_loss(
                        model, tokens, eval_thought_stacks, noise_scheduler,
                        device, epoch, config, args.mask_ratio, args.confidence_threshold, pad_token_id
                    )
                eval_losses.append(metrics)

        # Average evaluation metrics
        eval_metrics = {}
        for key in eval_losses[0].keys():
            values = []
            for l in eval_losses:
                val = l[key]
                if torch.is_tensor(val):
                    val = float(val.detach().cpu().item()) if val.numel() == 1 else val.detach().cpu().numpy()
                elif hasattr(val, 'item'):
                    val = float(val.item())
                values.append(val)
            eval_metrics[key] = np.mean(values)

        # Log evaluation metrics
        eval_step = (epoch + 1) * len(train_loader)
        logger.log_metrics(eval_metrics, eval_step, "eval", model)

        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        
        print(f"\n📈 Epoch {epoch + 1} Results:")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {epoch_metrics['avg_total_loss']:.4f}")
        print(f"    - Token Loss: {epoch_metrics['avg_token_loss']:.4f}")
        print(f"    - Masked Acc: {epoch_metrics['avg_masked_accuracy']:.3f}")
        print(f"    - Unmasked Acc: {epoch_metrics['avg_unmasked_accuracy']:.3f}")
        print(f"    - Thought RMS: {epoch_metrics['avg_thought_rms']:.3f}")
        if "avg_bidirectional_effectiveness" in epoch_metrics:
            print(f"    - Bidirectional Eff: {epoch_metrics['avg_bidirectional_effectiveness']:.3f}")
        print(f"  Eval Loss: {eval_metrics['total_loss']:.4f}")
        print(f"    - Masked Acc: {eval_metrics['masked_accuracy']:.3f}")
        print(f"    - Thought RMS: {eval_metrics['thought_rms']:.3f}")

        # Update plateau scheduler
        plateau_scheduler.step(eval_metrics["total_loss"])
        
        # Save best model and check early stopping
        if eval_metrics["total_loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["total_loss"]
            no_improvement_count = 0
            best_model_path = os.path.join(checkpoint_dir, "best_dream_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 New best model saved! (eval_loss: {best_eval_loss:.4f})")
        else:
            no_improvement_count += 1
            print(f"⚠️  No improvement for {no_improvement_count}/{early_stopping_patience} epochs")

            if no_improvement_count >= early_stopping_patience:
                print(f"🛑 Early stopping triggered!")
                break

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"dream_checkpoint_epoch_{epoch + 1}.pt")
        checkpoint_metadata = {
            "eval_metrics": eval_metrics,
            "epoch_metrics": epoch_metrics,
            "param_count": param_count,
            "best_eval_loss": best_eval_loss,
        }
        logger.save_checkpoint_with_metadata(
            model, optimizer, lr_scheduler, epoch + 1, checkpoint_metadata, checkpoint_path
        )
        print(f"💾 Checkpoint saved: {os.path.basename(checkpoint_path)}")

        torch.cuda.empty_cache()

    # Training complete
    total_training_time = time.time() - training_start_time

    print(f"\n🎉 DREAM model training complete!")
    print(f"  Total time: {total_training_time / 60:.1f} minutes")
    print(f"  Best eval loss: {best_eval_loss:.4f}")
    print(f"  Final model size: {param_count / 1e6:.1f}M parameters")

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "final_dream_model.pt")
    torch.save(model.state_dict(), final_model_path)

    # Save training summary
    summary = {
        "config": config,
        "training_summary": {
            "dataset_name": args.dataset,
            "train_samples": args.train_samples,
            "eval_samples": args.eval_samples,
            "epochs_completed": epoch + 1,
            "total_training_time_minutes": total_training_time / 60,
            "best_eval_loss": best_eval_loss,
            "parameter_count": param_count,
            "vocab_size": actual_vocab_size,
            "dream_config": {
                "mask_ratio": args.mask_ratio,
                "confidence_threshold": args.confidence_threshold,
                "bidirectional_attention": True,
                "self_supervised_thoughts": True,
                "adaptive_noise_scheduling": True,
            },
        },
        "final_metrics": eval_metrics,
    }

    summary_path = os.path.join(
        log_dir, f'dream_training_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"📊 Training summary saved: {summary_path}")
    print(f"✅ DREAM training completed successfully!")


if __name__ == "__main__":
    main()