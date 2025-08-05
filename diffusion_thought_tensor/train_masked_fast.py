"""
Fast-dLLM Masked Model Training with Hugging Face Datasets
Train the Masked Fast Diffusion Thought Tensor Model with real data
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
from typing import Dict, List, Optional, Tuple

# Import our models and utilities
from model.masked_fast_model import MaskedFastDiffusionModel, MaskingStrategy
from utils.noise_schedules import create_noise_scheduler
from utils.thought_analyzer import create_thought_analyzer
from data.huggingface_data_loader import (
    create_huggingface_data_loaders,
    SUPPORTED_DATASETS,
)


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
        if f.endswith(".pt") and "masked_checkpoint_epoch_" in f
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


def compute_masked_loss(
    model: MaskedFastDiffusionModel,
    batch: torch.Tensor,
    thought_stacks: torch.Tensor,
    noise_scheduler,
    device: torch.device,
    epoch: int = 0,
    config: Optional[Dict] = None,
    masking_ratio: float = 0.7,
    pad_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """Compute loss for masked Fast-dLLM training with thought evolution"""

    tokens = batch
    batch_size, seq_length = tokens.shape

    # Sample timesteps for diffusion
    timesteps = torch.randint(
        0, noise_scheduler.num_steps, (batch_size,), device=device
    )

    # Create masking pattern - mask a percentage of tokens
    # For training, we randomly mask tokens (not just at the end)
    token_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)

    for b in range(batch_size):
        # Find non-padding positions
        if pad_token_id is not None:
            # Only consider non-padding tokens for masking
            non_pad_positions = (tokens[b] != pad_token_id).nonzero(as_tuple=True)[0]
            if len(non_pad_positions) > 0:
                # Calculate how many non-padding tokens to mask
                non_pad_count = len(non_pad_positions)
                mask_count = int(non_pad_count * masking_ratio)
                if mask_count > 0:
                    # Randomly select from non-padding positions only
                    selected_indices = torch.randperm(non_pad_count, device=device)[:mask_count]
                    mask_positions = non_pad_positions[selected_indices]
                    token_mask[b, mask_positions] = True
        else:
            # Fallback to original logic if no pad_token_id provided
            mask_count = int(seq_length * masking_ratio)
            mask_positions = torch.randperm(seq_length, device=device)[:mask_count]
            token_mask[b, mask_positions] = True

    # Forward pass with masking
    pred_logits, new_thought = model(tokens, thought_stacks, timesteps, token_mask)

    # Evolve thought stack with new thought
    evolved_thought_stacks = model.thought_stack.push_thought(
        thought_stacks, new_thought
    )

    # Compute losses
    # 1. Primary token prediction loss (only on masked positions)
    masked_logits = pred_logits[token_mask]
    masked_targets = tokens[token_mask]

    if masked_logits.numel() > 0:
        token_loss = F.cross_entropy(
            masked_logits,
            masked_targets,
            reduction="mean",
            label_smoothing=0.1 if config else 0.0,
        )
    else:
        token_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # 2. Unmasking confidence loss (encourage confident predictions on unmasked)
    unmasked_logits = pred_logits[~token_mask]
    if unmasked_logits.numel() > 0:
        unmasked_probs = F.softmax(unmasked_logits, dim=-1)
        # Encourage high confidence (low entropy) on unmasked positions
        unmasked_entropy = -torch.sum(
            unmasked_probs * torch.log(unmasked_probs + 1e-8), dim=-1
        )
        confidence_loss = unmasked_entropy.mean()
    else:
        confidence_loss = torch.tensor(0.0, device=device)

    # 3. Thought consistency loss
    thought_consistency_loss = torch.tensor(0.0, device=device)
    if epoch > 2:  # Add after initial training
        # Encourage thought evolution smoothness
        thought_norm = torch.norm(new_thought, dim=1).mean()
        thought_consistency_loss = 0.01 * thought_norm

    # Combine losses
    total_loss = token_loss + 0.1 * confidence_loss

    # Compute metrics with comprehensive thought analysis
    with torch.no_grad():
        masked_accuracy = 0.0
        unmasked_accuracy = 0.0

        if masked_logits.numel() > 0:
            masked_preds = masked_logits.argmax(dim=-1)
            # Exclude padding tokens from masked accuracy calculation
            if pad_token_id is not None:
                non_padding_mask = masked_targets != pad_token_id
                if non_padding_mask.sum() > 0:
                    masked_accuracy = (
                        (masked_preds[non_padding_mask] == masked_targets[non_padding_mask])
                        .float().mean().item()
                    )
                else:
                    masked_accuracy = 0.0
            else:
                masked_accuracy = (masked_preds == masked_targets).float().mean().item()

        if unmasked_logits.numel() > 0:
            unmasked_targets = tokens[~token_mask]
            unmasked_preds = unmasked_logits.argmax(dim=-1)
            # Exclude padding tokens from unmasked accuracy calculation
            if pad_token_id is not None:
                non_padding_mask = unmasked_targets != pad_token_id
                if non_padding_mask.sum() > 0:
                    unmasked_accuracy = (
                        (unmasked_preds[non_padding_mask] == unmasked_targets[non_padding_mask])
                        .float().mean().item()
                    )
                else:
                    unmasked_accuracy = 0.0
            else:
                unmasked_accuracy = (
                    (unmasked_preds == unmasked_targets).float().mean().item()
                )

        perplexity = (
            torch.exp(token_loss).item() if torch.isfinite(token_loss) else 999.0
        )
        perplexity = min(perplexity, 1e4)

        # Enhanced thought metrics (from enhanced_model.py)
        thought_flat = new_thought.view(batch_size, -1)
        
        # 1. Thought activation strength (RMS instead of simple norm)
        thought_rms = torch.sqrt(torch.mean(thought_flat**2, dim=1)).mean().item()
        
        # 2. Thought sparsity (fraction of significant activations)
        threshold = 0.01 * thought_rms  # Adaptive threshold
        active_fraction = (torch.abs(thought_flat) > threshold).float().mean().item()
        
        # 3. Thought dynamic range utilization
        thought_min = thought_flat.min().item()
        thought_max = thought_flat.max().item()
        dynamic_range = thought_max - thought_min
        
        # 4. Thought consistency (how similar thoughts are within batch)
        if batch_size > 1:
            pairwise_sim = F.cosine_similarity(
                thought_flat.unsqueeze(1), thought_flat.unsqueeze(0), dim=2
            )
            # Average similarity excluding self-similarity diagonal
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=new_thought.device)
            thought_consistency = pairwise_sim[mask].mean().item()
        else:
            thought_consistency = 1.0
        
        # 5. Thought diversity metrics
        thought_diversity_loss = 0.0
        if batch_size > 1 and epoch > 2:
            # Normalize thoughts for better distance computation
            thought_normalized = F.normalize(thought_flat, dim=1)
            # Compute cosine similarities
            similarities = torch.mm(thought_normalized, thought_normalized.t())
            # Remove diagonal (self-similarities)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=new_thought.device)
            avg_similarity = similarities[mask].mean()
            thought_diversity_loss = avg_similarity.item()  # Higher similarity = less diversity
        
        # 6. Thought evolution smoothness (if we have previous thoughts)
        thought_evolution_smoothness = 0.0
        if epoch > 0:
            # Compare new thought with latest in stack
            latest_in_stack = thought_stacks[:, -1]  # Most recent thought in stack
            evolution_diff = F.mse_loss(new_thought, latest_in_stack, reduction='mean')
            thought_evolution_smoothness = evolution_diff.item()

        # Basic thought statistics
        thought_mean = new_thought.mean().item()
        thought_std = new_thought.std().item()
        thought_norm = torch.norm(new_thought, dim=1).mean().item()

        # Masking statistics - exclude padding tokens from ratio calculation
        if pad_token_id is not None:
            # Calculate mask ratio only for non-padding tokens
            non_pad_mask = tokens != pad_token_id
            if non_pad_mask.sum() > 0:
                # Count masked non-padding tokens / total non-padding tokens
                masked_non_pad = (token_mask & non_pad_mask).sum().float()
                total_non_pad = non_pad_mask.sum().float()
                mask_ratio_actual = (masked_non_pad / total_non_pad).item()
            else:
                mask_ratio_actual = 0.0
        else:
            # Fallback to original calculation if no pad_token_id
            mask_ratio_actual = token_mask.float().mean().item()
        
        # Store thought statistics for TensorBoard histograms
        # (This will be accessed by the logger for detailed visualizations)
        if hasattr(model, '_last_thought_stats') is False:
            model._last_thought_stats = {}
        model._last_thought_stats['thought_activations'] = new_thought.detach().cpu()

    metrics = {
        # Loss metrics
        "total_loss": total_loss.item() if torch.isfinite(total_loss) else 999.0,
        "token_loss": token_loss.item() if torch.isfinite(token_loss) else 999.0,
        "confidence_loss": (
            confidence_loss.item() if torch.isfinite(confidence_loss) else 0.0
        ),
        "thought_consistency_loss": (
            thought_consistency_loss.item()
            if torch.isfinite(thought_consistency_loss)
            else 0.0
        ),
        
        # Accuracy metrics
        "masked_accuracy": masked_accuracy,
        "unmasked_accuracy": unmasked_accuracy,
        "perplexity": perplexity,
        
        # Enhanced thought metrics
        "thought_rms": thought_rms,
        "thought_sparsity": 1.0 - active_fraction,  # Higher = more sparse/efficient
        "thought_dynamic_range": dynamic_range,
        "thought_consistency": thought_consistency,  # Batch-level consistency
        "thought_diversity_loss": thought_diversity_loss,  # Higher = less diverse
        "thought_evolution_smoothness": thought_evolution_smoothness,  # MSE with previous
        "thought_mean": thought_mean,
        "thought_std": thought_std,
        "thought_norm": thought_norm,
        "thought_min": thought_min,
        "thought_max": thought_max,
        "thought_active_fraction": active_fraction,
        
        # Masking statistics
        "mask_ratio": mask_ratio_actual,
        "masked_tokens": token_mask.sum().item(),
        "unmasked_tokens": (~token_mask).sum().item(),
        "total_non_pad_tokens": non_pad_mask.sum().item() if pad_token_id is not None else tokens.numel(),
    }

    return total_loss, metrics, evolved_thought_stacks


class MaskedTrainingLogger:
    """Enhanced logger for masked model training"""

    def __init__(self, log_dir, config, use_wandb=False, use_tensorboard=True):
        self.log_dir = log_dir
        self.config = config
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard

        # Setup logging backends
        if use_wandb:
            try:
                import wandb

                wandb.init(
                    project="masked-fast-dllm",
                    config=config,
                    name=f"masked_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                )
            except ImportError:
                print("Warning: wandb not available")
                self.use_wandb = False

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                os.makedirs(log_dir, exist_ok=True)
                tensorboard_dir = os.path.join(log_dir, "tensorboard")
                os.makedirs(tensorboard_dir, exist_ok=True)
                self.writer = SummaryWriter(tensorboard_dir)
                print(f"✅ TensorBoard logging enabled: {tensorboard_dir}")
            except ImportError:
                print("Warning: tensorboard not available")
                self.use_tensorboard = False

        self.metrics_history = []
        self.best_metrics = {}

    def log_metrics(self, metrics, step, phase="train", model=None):
        """Enhanced metrics logging with detailed TensorBoard visualizations"""
        metrics_with_meta = {
            "step": step,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }

        self.metrics_history.append(metrics_with_meta)

        # Log to wandb
        if self.use_wandb:
            try:
                import wandb
                wandb.log({f"{phase}/{k}": v for k, v in metrics.items()}, step=step)
            except ImportError:
                pass

        # Enhanced TensorBoard logging
        if self.use_tensorboard and hasattr(self, "writer"):
            try:
                # Group metrics by category for better organization
                loss_metrics = {}
                accuracy_metrics = {}
                thought_metrics = {}
                masking_metrics = {}
                training_metrics = {}
                
                for key, value in metrics.items():
                    if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                        continue
                    
                    # Categorize metrics
                    if "loss" in key:
                        loss_metrics[key] = value
                    elif "accuracy" in key or "perplexity" in key:
                        accuracy_metrics[key] = value
                    elif "thought" in key:
                        thought_metrics[key] = value
                    elif "mask" in key:
                        masking_metrics[key] = value
                    else:
                        training_metrics[key] = value
                
                # Log grouped metrics with custom tags
                for category, group_metrics in [
                    ("Losses", loss_metrics),
                    ("Accuracy", accuracy_metrics), 
                    ("Thoughts", thought_metrics),
                    ("Masking", masking_metrics),
                    ("Training", training_metrics)
                ]:
                    for key, value in group_metrics.items():
                        self.writer.add_scalar(f"{phase}/{category}/{key}", value, step)
                
                # Add custom composite metrics for analysis
                if phase == "train" and step % 100 == 0:  # Every 100 steps
                    # Thought health composite score
                    if all(k in metrics for k in ["thought_rms", "thought_sparsity", "thought_dynamic_range"]):
                        thought_health = (
                            metrics["thought_rms"] * 0.4 +
                            metrics["thought_sparsity"] * 0.3 +
                            (metrics["thought_dynamic_range"] / 10.0) * 0.3  # Normalize dynamic range
                        )
                        self.writer.add_scalar(f"{phase}/Composite/thought_health", thought_health, step)
                    
                    # Masking efficiency score
                    if all(k in metrics for k in ["masked_accuracy", "unmasked_accuracy", "confidence_loss"]):
                        masking_efficiency = (
                            metrics["masked_accuracy"] * 0.5 +
                            metrics["unmasked_accuracy"] * 0.3 +
                            max(0, 1.0 - metrics["confidence_loss"] / 10.0) * 0.2  # Inverse confidence loss
                        )
                        self.writer.add_scalar(f"{phase}/Composite/masking_efficiency", masking_efficiency, step)
                
                # Create histograms for key thought metrics (less frequent to avoid overhead)
                if phase == "train" and step % 200 == 0 and model is not None:
                    try:
                        # Sample a batch to get thought distributions
                        with torch.no_grad():
                            # Get current thought statistics from last computation
                            if hasattr(model, '_last_thought_stats'):
                                stats = model._last_thought_stats
                                if 'thought_activations' in stats:
                                    self.writer.add_histogram(
                                        f"{phase}/Distributions/thought_activations", 
                                        stats['thought_activations'], 
                                        step
                                    )
                    except Exception as e:
                        # Don't fail training if histogram logging fails
                        pass
                
                # Log learning curves for key metrics
                if step > 100:  # After some initial steps
                    recent_steps = 50
                    recent_metrics = [m for m in self.metrics_history[-recent_steps:] if m["phase"] == phase]
                    
                    if len(recent_metrics) >= 10:  # Need some history
                        # Calculate trends for key metrics
                        for metric_name in ["total_loss", "thought_rms", "masked_accuracy"]:
                            if metric_name in recent_metrics[-1]:
                                values = [m[metric_name] for m in recent_metrics if metric_name in m]
                                if len(values) >= 10:
                                    # Simple trend calculation (slope of last 10 points)
                                    x = np.arange(len(values[-10:]))
                                    y = np.array(values[-10:])
                                    if len(x) > 1 and np.std(x) > 0:
                                        slope = np.corrcoef(x, y)[0, 1] if not np.isnan(np.corrcoef(x, y)[0, 1]) else 0
                                        self.writer.add_scalar(f"{phase}/Trends/{metric_name}_trend", slope, step)

                self.writer.flush()
                
            except Exception as e:
                print(f"⚠️  TensorBoard logging error: {e}")

        # Save to file
        log_file = os.path.join(self.log_dir, f"{phase}_metrics.jsonl")
        with open(log_file, "a") as f:
            json.dump(metrics_with_meta, f)
            f.write("\n")

        # Update best metrics
        for key, value in metrics.items():
            if key.endswith("loss"):
                if key not in self.best_metrics or value < self.best_metrics[key]:
                    self.best_metrics[key] = value
                    self.best_metrics[f"{key}_step"] = step

    def save_checkpoint_with_metadata(
        self, model, optimizer, scheduler, epoch, metrics, checkpoint_path
    ):
        """Save training checkpoint with metadata"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics,
            "best_metrics": self.best_metrics,
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
        }

        torch.save(checkpoint, checkpoint_path)

        if self.use_wandb:
            try:
                import wandb

                wandb.save(checkpoint_path)
            except ImportError:
                pass

        return checkpoint_path


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


def main():
    """Main function for masked Fast-dLLM training"""

    import argparse

    parser = argparse.ArgumentParser(description="Train Masked Fast-dLLM Model")
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
        "--auto_resume",
        action="store_true",
        help="Automatically resume from latest checkpoint",
    )
    parser.add_argument(
        "--mask_ratio", type=float, default=0.7, help="Token masking ratio"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.8,
        help="Confidence threshold for unmasking",
    )

    args = parser.parse_args()

    print(f"🎭 Starting Masked Fast-dLLM training with {args.dataset}")
    print(f"   Mask ratio: {args.mask_ratio}")
    print(f"   Confidence threshold: {args.confidence_threshold}")

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override epochs
    config["training"]["num_epochs"] = args.epochs

    # Setup directories
    checkpoint_dir = config["checkpointing"]["checkpoint_dir"].replace(
        "enhanced_", "masked_"
    )
    log_dir = config["checkpointing"]["log_dir"].replace("enhanced_", "masked_")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Device setup
    device = torch.device(config["hardware"]["device"])
    print(f"Using device: {device}")

    # Get pad_token_id from tokenizer (GPT2 uses eos_token as pad_token)
    # For GPT2, the eos_token_id is 50256
    pad_token_id = 50256  # GPT2's eos_token_id used as pad_token

    # Create masking strategy
    masking_strategy = MaskingStrategy(
        mask_ratio=args.mask_ratio,
        confidence_threshold=args.confidence_threshold,
        progressive_unmasking=True,
        block_size=1,
    )

    # Initialize masked Fast-dLLM model
    print("Initializing Masked Fast-dLLM Model...")
    model = MaskedFastDiffusionModel(
        vocab_size=config["model"].get("vocab_size", 50257),
        embed_dim=config["model"]["embed_dim"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        max_seq_length=config["model"]["max_seq_length"],
        max_new_tokens=config["model"].get("max_new_tokens", 1024),
        thought_dims=tuple(config["thought_tensor"]["input_dims"]),
        thought_stack_size=config["thought_tensor"]["stack_size"],
        dropout=config["model"]["dropout"],
        masking_strategy=masking_strategy,
    )
    model = model.to(device).float()

    param_count = count_parameters(model)
    print(
        f"Masked Fast-dLLM model initialized with {param_count / 1e6:.1f}M parameters"
    )

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
            print(
                f"Updating model vocab size: {model.vocab_size} -> {actual_vocab_size}"
            )
            # Resize token embeddings and output projection
            old_embeddings = model.token_embeddings
            model.token_embeddings = nn.Embedding(
                actual_vocab_size, model.embed_dim
            ).to(device)
            # Copy existing weights for common tokens
            min_vocab = min(old_embeddings.num_embeddings, actual_vocab_size)
            model.token_embeddings.weight.data[:min_vocab] = old_embeddings.weight.data[
                :min_vocab
            ]

            # Update output projection
            model.output_projection = nn.Linear(model.embed_dim, actual_vocab_size).to(
                device
            )
            model.vocab_size = actual_vocab_size

    except Exception as e:
        print(f"Error loading Hugging Face dataset: {e}")
        print("Please install: pip install datasets transformers")
        return

    # Initialize training components
    noise_scheduler = create_noise_scheduler(config["diffusion"])
    if hasattr(noise_scheduler, "sqrt_alphas_cumprod"):
        noise_scheduler.sqrt_alphas_cumprod = noise_scheduler.sqrt_alphas_cumprod.to(
            device
        )
        noise_scheduler.sqrt_one_minus_alphas_cumprod = (
            noise_scheduler.sqrt_one_minus_alphas_cumprod.to(device)
        )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        betas=config["optimizer"]["betas"],
        weight_decay=float(config["optimizer"]["weight_decay"]),
        eps=float(config["optimizer"]["eps"]),
    )

    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
    
    # Primary scheduler with warm restarts
    cosine_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(config["training"]["warmup_steps"]),
        T_mult=2,
        eta_min=float(config["training"]["learning_rate"]) * 0.01,
    )
    
    # Plateau detection scheduler for adaptive reduction
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
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
    
    # Progressive masking configuration
    initial_mask_ratio = args.mask_ratio
    final_mask_ratio = args.mask_ratio * 0.3  # Reduce to 30% of initial by end
    mask_decay_start_epoch = max(1, num_epochs // 3)  # Start decay after 1/3 of training

    # Initialize mixed precision
    scaler = (
        GradScaler("cuda")
        if config.get("hardware", {}).get("mixed_precision", True)
        else None
    )
    use_amp = scaler is not None

    # Initialize logger
    use_wandb = config.get("logging", {}).get("use_wandb", False)
    use_tensorboard = config.get("logging", {}).get("use_tensorboard", True)
    logger = MaskedTrainingLogger(log_dir, config, use_wandb, use_tensorboard)

    # Training info
    print(f"\n🎭 Starting Masked Fast-dLLM training:")
    print(f"  Model: {param_count / 1e6:.1f}M parameters")
    print(f"  Dataset: {args.dataset}")
    print(f"  Vocabulary: {actual_vocab_size} tokens")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Evaluation samples: {len(eval_loader.dataset)}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Masking ratio: {args.mask_ratio}")
    print(f"  Confidence threshold: {args.confidence_threshold}")
    print(f"  Mixed Precision: {'Enabled' if use_amp else 'Disabled'}")

    training_start_time = time.time()
    no_improvement_count = 0

    # Persistent thought stack configuration
    stack_reset_frequency = config.get("training", {}).get("stack_reset_frequency", 10)
    persistent_thought_stacks = None
    batch_count = 0
    print(f"  Thought Stack Persistence: Reset every {stack_reset_frequency} batches")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        # Training phase
        model.train()
        epoch_losses = []

        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

        # Calculate dynamic masking ratio for this epoch
        if epoch >= mask_decay_start_epoch:
            progress = (epoch - mask_decay_start_epoch) / max(1, num_epochs - mask_decay_start_epoch)
            current_mask_ratio = initial_mask_ratio - (initial_mask_ratio - final_mask_ratio) * progress
        else:
            current_mask_ratio = initial_mask_ratio
        
        # Dynamic confidence threshold - increase as training progresses
        progress_ratio = epoch / max(1, num_epochs - 1)
        current_confidence_threshold = args.confidence_threshold + (0.9 - args.confidence_threshold) * progress_ratio
        
        print(f"Using masking ratio: {current_mask_ratio:.3f}, confidence threshold: {current_confidence_threshold:.3f}")

        for batch_idx, batch in enumerate(train_pbar):
            tokens = batch.to(device, non_blocking=True)
            batch_size = tokens.shape[0]

            # Handle persistent thought stacks with reset frequency
            if (
                batch_count % stack_reset_frequency == 0
                or persistent_thought_stacks is None
                or persistent_thought_stacks.shape[0] != batch_size
            ):
                # Reset or initialize thought stacks
                current_thought_stacks = model.initialize_thought_stack(
                    batch_size, device
                )
                # if batch_count % stack_reset_frequency == 0 and batch_count > 0:
                #     print(f"    🔄 Resetting thought stacks at batch {batch_count}")
            else:
                # Use persistent stacks from previous batch
                current_thought_stacks = persistent_thought_stacks

            # Update masking strategy for this batch
            model.masking_strategy.confidence_threshold = current_confidence_threshold

            # Compute loss with mixed precision
            if use_amp:
                with autocast("cuda"):
                    loss, metrics, evolved_stacks = compute_masked_loss(
                        model,
                        tokens,
                        current_thought_stacks,
                        noise_scheduler,
                        device,
                        epoch,
                        config,
                        current_mask_ratio,
                        pad_token_id,
                    )
                    loss = loss / gradient_accumulation_steps

                scaler.scale(loss).backward()
            else:
                loss, metrics, evolved_stacks = compute_masked_loss(
                    model,
                    tokens,
                    current_thought_stacks,
                    noise_scheduler,
                    device,
                    epoch,
                    config,
                    args.mask_ratio,
                    pad_token_id,
                )
                loss = loss / gradient_accumulation_steps
                loss.backward()

            # Store evolved stacks for next batch (detached to prevent gradient accumulation)
            persistent_thought_stacks = evolved_stacks.detach()
            batch_count += 1

            # Optimizer step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    
                    # Add gradient noise for regularization when plateauing
                    if no_improvement_count > 2:
                        noise_scale = 0.01 * (no_improvement_count - 2)
                        for param in model.parameters():
                            if param.grad is not None:
                                noise = torch.randn_like(param.grad) * noise_scale
                                param.grad.add_(noise)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    
                    # Add gradient noise for regularization when plateauing
                    if no_improvement_count > 2:
                        noise_scale = 0.01 * (no_improvement_count - 2)
                        for param in model.parameters():
                            if param.grad is not None:
                                noise = torch.randn_like(param.grad) * noise_scale
                                param.grad.add_(noise)
                    
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()

                metrics["grad_norm"] = grad_norm.item()
                metrics["learning_rate"] = lr_scheduler.get_last_lr()[0]

            epoch_losses.append(metrics)

            # Log metrics with enhanced TensorBoard support
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % 50 == 0:
                logger.log_metrics(metrics, global_step, "train", model)

            # Update progress bar with enhanced thought metrics
            progress_info = {
                "loss": f"{metrics['total_loss']:.4f}",
                "tok_loss": f"{metrics['token_loss']:.4f}",
                "conf_loss": f"{metrics['confidence_loss']:.4f}",
                "m_acc": f"{metrics['masked_accuracy']:.3f}",
                "u_acc": f"{metrics['unmasked_accuracy']:.3f}",
                "t_rms": f"{metrics['thought_rms']:.3f}",
                "t_sparse": f"{metrics['thought_sparsity']:.2f}",
                "t_div": f"{metrics['thought_diversity_loss']:.3f}",
                "mask": f"{metrics['mask_ratio']:.2f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
            }
            train_pbar.set_postfix(progress_info)

            # Memory cleanup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

        # Compute epoch averages
        epoch_metrics = {}
        for key in epoch_losses[0].keys():
            epoch_metrics[f"avg_{key}"] = np.mean([l[key] for l in epoch_losses])

        # Evaluation phase
        print("\n📊 Evaluating masked model...")
        model.eval()
        eval_losses = []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                tokens = batch.to(device, non_blocking=True)
                batch_size = tokens.shape[0]

                # Initialize fresh thought stacks for evaluation
                eval_thought_stacks = model.initialize_thought_stack(batch_size, device)

                if use_amp:
                    with autocast("cuda"):
                        loss, metrics, _ = compute_masked_loss(
                            model,
                            tokens,
                            eval_thought_stacks,
                            noise_scheduler,
                            device,
                            epoch,
                            config,
                            args.mask_ratio,
                            pad_token_id,
                        )
                else:
                    loss, metrics, _ = compute_masked_loss(
                        model,
                        tokens,
                        eval_thought_stacks,
                        noise_scheduler,
                        device,
                        epoch,
                        config,
                        args.mask_ratio,  # Use original ratio for evaluation consistency
                        pad_token_id,
                    )
                eval_losses.append(metrics)

        # Average evaluation metrics
        eval_metrics = {}
        for key in eval_losses[0].keys():
            eval_metrics[key] = np.mean([l[key] for l in eval_losses])

        # Log evaluation metrics
        eval_step = (epoch + 1) * len(train_loader)
        logger.log_metrics(eval_metrics, eval_step, "eval", model)

        # Epoch summary
        epoch_time = time.time() - epoch_start_time

        print(f"\n📈 Epoch {epoch + 1} Results:")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {epoch_metrics['avg_total_loss']:.4f}")
        print(f"    - Token Loss: {epoch_metrics['avg_token_loss']:.4f}")
        print(f"    - Confidence Loss: {epoch_metrics['avg_confidence_loss']:.4f}")
        print(f"    - Masked Accuracy: {epoch_metrics['avg_masked_accuracy']:.3f}")
        print(f"    - Unmasked Accuracy: {epoch_metrics['avg_unmasked_accuracy']:.3f}")
        print(f"  Eval Loss: {eval_metrics['total_loss']:.4f}")
        print(f"    - Token Loss: {eval_metrics['token_loss']:.4f}")
        print(f"    - Masked Accuracy: {eval_metrics['masked_accuracy']:.3f}")
        print(f"    - Unmasked Accuracy: {eval_metrics['unmasked_accuracy']:.3f}")
        print(f"  Thought Metrics:")
        print(f"    - RMS: {epoch_metrics.get('avg_thought_rms', 0):.3f}")
        print(f"    - Sparsity: {epoch_metrics.get('avg_thought_sparsity', 0):.3f}")
        print(f"    - Dynamic Range: {epoch_metrics.get('avg_thought_dynamic_range', 0):.2f}")
        print(f"    - Consistency: {epoch_metrics.get('avg_thought_consistency', 0):.3f}")
        print(f"    - Diversity Loss: {epoch_metrics.get('avg_thought_diversity_loss', 0):.3f}")
        print(f"  Learning Rate: {lr_scheduler.get_last_lr()[0]:.2e}")

        # Update plateau scheduler
        plateau_scheduler.step(eval_metrics["total_loss"])
        
        # Save best model and check early stopping
        if eval_metrics["total_loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["total_loss"]
            no_improvement_count = 0
            best_model_path = os.path.join(checkpoint_dir, "best_masked_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 New best model saved! (eval_loss: {best_eval_loss:.4f})")
        else:
            no_improvement_count += 1
            print(
                f"⚠️  No improvement for {no_improvement_count}/{early_stopping_patience} epochs"
            )

            if no_improvement_count >= early_stopping_patience:
                print(f"🛑 Early stopping triggered!")
                break

        # Save checkpoint
        checkpoint_path = os.path.join(
            checkpoint_dir, f"masked_checkpoint_epoch_{epoch + 1}.pt"
        )
        checkpoint_metadata = {
            "eval_metrics": eval_metrics,
            "epoch_metrics": epoch_metrics,
            "param_count": param_count,
            "best_eval_loss": best_eval_loss,
        }
        logger.save_checkpoint_with_metadata(
            model,
            optimizer,
            lr_scheduler,
            epoch + 1,
            checkpoint_metadata,
            checkpoint_path,
        )
        print(f"💾 Checkpoint saved: {os.path.basename(checkpoint_path)}")

        torch.cuda.empty_cache()

    # Training complete
    total_training_time = time.time() - training_start_time

    print(f"\n🎉 Masked Fast-dLLM training complete!")
    print(f"  Total time: {total_training_time / 60:.1f} minutes")
    print(f"  Best eval loss: {best_eval_loss:.4f}")
    print(f"  Final model size: {param_count / 1e6:.1f}M parameters")

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "final_masked_model.pt")
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
            "masking_strategy": {
                "mask_ratio": args.mask_ratio,
                "confidence_threshold": args.confidence_threshold,
            },
        },
        "final_metrics": eval_metrics,
    }

    summary_path = os.path.join(
        log_dir,
        f'masked_training_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"📊 Training summary saved: {summary_path}")
    print(f"✅ Masked Fast-dLLM training completed successfully!")


if __name__ == "__main__":
    main()
