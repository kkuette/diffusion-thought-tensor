"""
Enhanced Model Training with Hugging Face Datasets
Train the Enhanced Diffusion Thought Tensor Model with real data
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
from model.enhanced_model import EnhancedDiffusionThoughtModel
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
        if f.endswith(".pt") and "enhanced_checkpoint_epoch_" in f
    ]
    if not checkpoint_files:
        return None

    # Extract epoch numbers and find the latest
    epochs = []
    for f in checkpoint_files:
        try:
            # Extract epoch number from filename like "enhanced_checkpoint_epoch_5.pt"
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

    # Get starting epoch
    start_epoch = checkpoint.get("epoch", 0)

    # Get best metrics
    best_metrics = checkpoint.get("best_metrics", {})
    best_eval_loss = best_metrics.get("total_loss", float("inf"))

    print(f"✅ Checkpoint loaded successfully!")
    print(f"   Resuming from epoch: {start_epoch}")
    print(f"   Best eval loss so far: {best_eval_loss:.4f}")

    return start_epoch, best_eval_loss, checkpoint.get("metrics", {})


class EnhancedTrainingLogger:
    """Comprehensive training logger with multiple backends"""

    def __init__(self, log_dir, config, use_wandb=False, use_tensorboard=True):
        self.log_dir = log_dir
        self.config = config

        # Setup logging backends
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard

        if use_wandb:
            try:
                import wandb

                wandb.init(
                    project="diffusion-thought-tensor-enhanced",
                    config=config,
                    name=f"enhanced_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                )
            except ImportError:
                print("Warning: wandb not available, skipping wandb logging")
                self.use_wandb = False

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                # Ensure log directory exists
                os.makedirs(log_dir, exist_ok=True)
                tensorboard_dir = os.path.join(log_dir, "tensorboard")
                os.makedirs(tensorboard_dir, exist_ok=True)
                self.writer = SummaryWriter(tensorboard_dir)
                print(f"✅ TensorBoard logging enabled: {tensorboard_dir}")
            except ImportError:
                print(
                    "⚠️  Warning: tensorboard not available, skipping tensorboard logging"
                )
                self.use_tensorboard = False
            except Exception as e:
                print(f"⚠️  Warning: Failed to initialize TensorBoard: {e}")
                self.use_tensorboard = False

        # Metrics storage
        self.metrics_history = []
        self.best_metrics = {}

    def log_metrics(self, metrics, step, phase="train"):
        """Log metrics to all backends"""
        # Add metadata
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

        # Log to tensorboard with error handling
        if self.use_tensorboard and hasattr(self, "writer"):
            try:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and not (
                        np.isnan(value) or np.isinf(value)
                    ):
                        self.writer.add_scalar(f"{phase}/{key}", value, step)
                # Flush immediately for real-time viewing
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

    def log_model_analysis(self, model, step):
        """Log detailed model analysis"""
        analysis = {
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }

        # Compute gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        analysis["gradient_norm"] = total_norm ** (1.0 / 2)

        # Layer-wise statistics (sample some key layers)
        key_modules = ["token_embeddings", "thought_stack_encoder", "output_projection"]
        for name, module in model.named_modules():
            if any(key in name for key in key_modules) and isinstance(
                module, nn.Linear
            ):
                weight = module.weight.data
                analysis[f"weight_norm/{name}"] = weight.norm().item()
                analysis[f"weight_mean/{name}"] = weight.mean().item()
                analysis[f"weight_std/{name}"] = weight.std().item()

        self.log_metrics(analysis, step, "model")

    def save_checkpoint_with_metadata(
        self, model, optimizer, scheduler, epoch, metrics, checkpoint_path
    ):
        """Save training checkpoint with comprehensive metadata"""
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

        # Log to wandb if available
        if self.use_wandb:
            try:
                import wandb

                wandb.save(checkpoint_path)
            except ImportError:
                pass

        return checkpoint_path

    def create_training_report(self):
        """Generate comprehensive training report"""
        report = {
            "config": self.config,
            "best_metrics": self.best_metrics,
            "training_duration": len(self.metrics_history),
            "final_metrics": self.metrics_history[-1] if self.metrics_history else {},
        }

        # Compute training statistics
        if self.metrics_history:
            train_metrics = [m for m in self.metrics_history if m["phase"] == "train"]
            eval_metrics = [m for m in self.metrics_history if m["phase"] == "eval"]

            if train_metrics:
                report["training_stats"] = {
                    "total_steps": len(train_metrics),
                    "avg_train_loss": np.mean(
                        [m.get("total_loss", 0) for m in train_metrics]
                    ),
                    "avg_eval_loss": (
                        np.mean([m.get("total_loss", 0) for m in eval_metrics])
                        if eval_metrics
                        else None
                    ),
                }

        report_path = os.path.join(self.log_dir, "training_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report


class MemoryMonitor:
    """Simple memory monitoring utility"""

    def __init__(self, target_memory_gb=16):
        self.target_memory_gb = target_memory_gb
        self.peak_memory_gb = 0
        self.memory_warnings = 0

    def get_current_memory_gb(self):
        """Get current GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        return 0

    def get_max_memory_gb(self):
        """Get peak GPU memory usage in GB"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**3
        return 0

    def check_memory_usage(self, step=None):
        """Check current memory usage and issue warnings if needed"""
        current_memory = self.get_current_memory_gb()
        max_memory = self.get_max_memory_gb()

        # Update peak tracking
        self.peak_memory_gb = max(self.peak_memory_gb, max_memory)

        # Memory warning
        if current_memory > self.target_memory_gb * 0.9:
            self.memory_warnings += 1
            print(
                f"⚠️  High memory usage: {current_memory:.1f}GB/{self.target_memory_gb}GB"
            )
            return True

        return False

    def cleanup_memory(self):
        """Perform memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Force garbage collection
        import gc

        gc.collect()

        return self.get_current_memory_gb()

    def get_memory_summary(self):
        """Get memory usage summary"""
        return {
            "current_memory_gb": self.get_current_memory_gb(),
            "max_memory_gb": self.get_max_memory_gb(),
            "peak_memory_gb": self.peak_memory_gb,
            "memory_warnings": self.memory_warnings,
        }


def compute_enhanced_loss_v2(
    model: EnhancedDiffusionThoughtModel,
    batch: Tuple[torch.Tensor, torch.Tensor],
    noise_scheduler,
    device: torch.device,
    epoch: int = 0,
    config: Optional[Dict] = None,
    thought_analyzer=None,
    previous_thoughts: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """Enhanced loss with multiple components for better training"""

    tokens, thought_stacks = batch
    batch_size = tokens.shape[0]

    # Sample timesteps
    timesteps = torch.randint(
        0, noise_scheduler.num_steps, (batch_size,), device=device
    )

    # Forward pass
    pred_logits, new_thought = model(tokens, thought_stacks, timesteps)
    evolved_stacks = model.evolve_thought_stack(thought_stacks, new_thought)

    # Primary language modeling loss with improved label smoothing
    pad_token_id = config.get("pad_token_id", 0) if config else 0
    label_smoothing = (
        config.get("training", {}).get("label_smoothing", 0.1) if config else 0.1
    )
    token_loss = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.shape[-1]),
        tokens.reshape(-1),
        ignore_index=pad_token_id,
        reduction="mean",
        label_smoothing=label_smoothing,
    )

    # Calculate perplexity
    with torch.no_grad():
        perplexity = torch.exp(token_loss).item()
        perplexity = min(perplexity, 1e4)

    # Auxiliary losses
    auxiliary_losses = {}

    # 1. Thought consistency loss (temporal coherence)
    if epoch > 2:  # Only after initial training
        thought_diff = new_thought - thought_stacks[:, -1]
        thought_consistency = torch.mean(thought_diff.pow(2))
        auxiliary_losses["thought_consistency"] = 0.1 * thought_consistency

    # 2. Thought diversity loss (prevent collapse) - IMPROVED
    if epoch > 2:  # Start earlier
        # Compute pairwise distances between thoughts in batch
        thought_flat = new_thought.view(batch_size, -1)
        if batch_size > 1:
            # Normalize thoughts for better distance computation
            thought_normalized = F.normalize(thought_flat, dim=1)
            # Compute cosine similarities
            similarities = torch.mm(thought_normalized, thought_normalized.t())
            # Remove diagonal (self-similarities)
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=new_thought.device)
            avg_similarity = similarities[mask].mean()
            # Penalize high similarities (encourage diversity)
            diversity_loss = avg_similarity  # Higher similarity = higher loss
            diversity_weight = config.get("thought_tensor", {}).get(
                "diversity_loss_weight", 0.1
            )
            auxiliary_losses["thought_diversity"] = diversity_weight * diversity_loss

    # 3. Thought sparsity (encourage efficient representations)
    if config and config.get("use_sparsity", False):
        sparsity_loss = torch.mean(torch.abs(new_thought))
        auxiliary_losses["thought_sparsity"] = 0.01 * sparsity_loss

    # Curriculum learning weight
    curriculum_weight = min(1.0, epoch / 10.0)

    # Combine losses
    total_loss = token_loss
    for name, loss in auxiliary_losses.items():
        total_loss = total_loss + curriculum_weight * loss

    # Comprehensive metrics with meaningful thought analysis
    with torch.no_grad():
        # More meaningful thought metrics
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

    metrics = {
        "total_loss": total_loss.item() if torch.isfinite(total_loss) else 999.0,
        "token_loss": token_loss.item() if torch.isfinite(token_loss) else 999.0,
        "perplexity": perplexity,
        "thought_rms": thought_rms,  # Root Mean Square - better than simple norm
        "thought_sparsity": 1.0 - active_fraction,  # Higher = more sparse/efficient
        "thought_dynamic_range": dynamic_range,  # How much of activation space is used
        "thought_consistency": thought_consistency,  # Batch-level consistency
        "thought_mean": new_thought.mean().item(),
        "thought_std": new_thought.std().item(),
        "curriculum_weight": curriculum_weight,
    }

    # Add auxiliary losses to metrics
    for name, loss in auxiliary_losses.items():
        metrics[name] = loss.item()

    # Compute advanced thought metrics if analyzer is provided
    if thought_analyzer is not None:
        try:
            advanced_metrics = thought_analyzer.compute_all_metrics(
                current_thoughts=new_thought,
                previous_thoughts=previous_thoughts,
                predictions=pred_logits,
                targets=tokens,
            )
            # Add prefix to avoid naming conflicts
            for key, value in advanced_metrics.items():
                metrics[f"advanced_{key}"] = value
        except Exception as e:
            # Don't fail training if metrics computation fails
            print(f"Warning: Advanced metrics computation failed: {e}")

    # Check for NaN/Inf in total loss
    if not torch.isfinite(total_loss):
        print(f"Warning: Non-finite loss detected. Setting to high value.")
        total_loss = torch.tensor(999.0, device=device, requires_grad=True)

    return total_loss, metrics, evolved_stacks


# Keep the original function for compatibility, but route to new one
def compute_enhanced_loss(
    model: EnhancedDiffusionThoughtModel,
    batch: Tuple[torch.Tensor, torch.Tensor],
    noise_scheduler,
    device: torch.device,
    epoch: int = 0,
    config: Optional[Dict] = None,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """Wrapper for backwards compatibility"""
    return compute_enhanced_loss_v2(
        model, batch, noise_scheduler, device, epoch, config, None, None
    )


def main():
    """Main function for enhanced model training"""

    import argparse

    parser = argparse.ArgumentParser(
        description="Train Enhanced Diffusion Thought Tensor Model"
    )
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

    args = parser.parse_args()

    if args.resume:
        print(f"🔄 Resuming enhanced model training with {args.dataset}")
        print(f"   Resume checkpoint: {args.resume}")
    elif args.auto_resume:
        print(f"🔄 Auto-resuming enhanced model training with {args.dataset}")
    else:
        print(f"🚀 Starting enhanced model training with {args.dataset}")

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override epochs
    config["training"]["num_epochs"] = args.epochs

    # Setup directories
    checkpoint_dir = config["checkpointing"]["checkpoint_dir"]
    log_dir = config["checkpointing"]["log_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Device setup
    device = torch.device(config["hardware"]["device"])
    print(f"Using device: {device}")

    # Initialize enhanced model with regularization
    print("Initializing Enhanced Diffusion Thought Model...")
    use_gradient_checkpointing = config.get("hardware", {}).get(
        "gradient_checkpointing", True
    )
    model = EnhancedDiffusionThoughtModel(
        vocab_size=config["model"].get("vocab_size", 50257),
        embed_dim=config["model"]["embed_dim"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        max_seq_length=config["model"]["max_seq_length"],
        max_new_tokens=config["model"].get("max_new_tokens", 1024),
        thought_dims=tuple(config["thought_tensor"]["input_dims"]),
        thought_stack_size=config["thought_tensor"]["stack_size"],
        dropout=config["model"]["dropout"],
        thought_dropout=config["thought_tensor"].get("dropout", 0.15),
        noise_injection_std=config["thought_tensor"].get("noise_injection_std", 0.01),
        use_gradient_checkpointing=use_gradient_checkpointing,
    )
    model = model.to(device).float()  # Ensure float32

    param_count = count_parameters(model)
    print(f"Enhanced model initialized with {param_count / 1e6:.1f}M parameters")

    # Check if Hugging Face datasets are available
    try:
        # Create Hugging Face data loaders
        print(f"Creating Hugging Face data loaders...")
        train_loader, eval_loader = create_huggingface_data_loaders(
            config,
            dataset_name=args.dataset,
            train_samples=args.train_samples,
            eval_samples=args.eval_samples,
            tokenizer_name="gpt2",
        )

        # Update vocab size in model if needed
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
        print("Please install required packages: pip install datasets transformers")
        return

    # Initialize training components
    noise_scheduler = create_noise_scheduler(config["diffusion"])
    # Move scheduler to device
    if hasattr(noise_scheduler, "sqrt_alphas_cumprod"):
        noise_scheduler.sqrt_alphas_cumprod = noise_scheduler.sqrt_alphas_cumprod.to(
            device
        )
        noise_scheduler.sqrt_one_minus_alphas_cumprod = (
            noise_scheduler.sqrt_one_minus_alphas_cumprod.to(device)
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        betas=config["optimizer"]["betas"],
        weight_decay=float(config["optimizer"]["weight_decay"]),
        eps=float(config["optimizer"]["eps"]),
    )

    # Scheduler
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

    lr_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(config["training"]["warmup_steps"]),
        T_mult=2,
        eta_min=float(config["training"]["learning_rate"]) * 0.01,
    )

    # Checkpoint loading logic
    start_epoch = 0
    best_eval_loss = float("inf")
    resume_checkpoint = None

    # Determine checkpoint to load
    if args.resume:
        resume_checkpoint = args.resume
    elif args.auto_resume:
        resume_checkpoint = find_latest_checkpoint(checkpoint_dir)

    # Load checkpoint if available
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        try:
            start_epoch, best_eval_loss, _ = load_checkpoint(
                resume_checkpoint, model, optimizer, lr_scheduler, device
            )
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            print("Starting from scratch...")
            start_epoch = 0
            best_eval_loss = float("inf")
    elif resume_checkpoint:
        print(f"⚠️  Checkpoint not found: {resume_checkpoint}")
        print("Starting from scratch...")

    # Training loop configuration
    num_epochs = config["training"]["num_epochs"]
    gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
    max_grad_norm = float(config["training"]["max_grad_norm"])

    # Early stopping configuration
    early_stopping_patience = config["training"].get("early_stopping_patience", 3)
    no_improvement_count = 0
    print(f"  Early stopping patience: {early_stopping_patience} epochs")

    # Adjust training if resuming
    remaining_epochs = num_epochs - start_epoch
    if start_epoch > 0:
        print(f"\n🔄 Resuming enhanced model training:")
        print(f"  Starting from epoch: {start_epoch + 1}")
        print(f"  Remaining epochs: {remaining_epochs}")
        print(f"  Current best eval loss: {best_eval_loss:.4f}")
    else:
        print(f"\n🚀 Starting enhanced model training:")
    print(f"  Model: {param_count / 1e6:.1f}M parameters")
    print(f"  Dataset: {args.dataset}")
    print(f"  Vocabulary: {actual_vocab_size} tokens")
    print(f"  Training samples: {len(train_loader.dataset)}")
    print(f"  Evaluation samples: {len(eval_loader.dataset)}")
    print(f"  Epochs: {num_epochs}")
    print(
        f"  Batch size: {config['training']['batch_size']} (Effective: {config['training']['batch_size'] * gradient_accumulation_steps})"
    )
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Thought stack size: {config['thought_tensor']['stack_size']}")

    # Initialize mixed precision scaler
    scaler = (
        GradScaler("cuda")
        if config.get("hardware", {}).get("mixed_precision", True)
        else None
    )
    use_amp = scaler is not None

    # Initialize memory monitor with more conservative limits
    memory_monitor = MemoryMonitor(
        target_memory_gb=16
    )  # Conservative limit for RTX 3090

    # Initialize enhanced logger
    use_wandb = config.get("logging", {}).get("use_wandb", False)
    use_tensorboard = config.get("logging", {}).get("use_tensorboard", True)
    enhanced_logger = EnhancedTrainingLogger(
        log_dir, config, use_wandb, use_tensorboard
    )

    # Initialize thought analyzer for advanced metrics
    use_advanced_metrics = config.get("logging", {}).get(
        "use_advanced_thought_metrics", True
    )
    thought_analyzer = create_thought_analyzer(config) if use_advanced_metrics else None

    training_start_time = time.time()

    print(f"  Mixed Precision: {'Enabled' if use_amp else 'Disabled'}")
    print(
        f"  Gradient Checkpointing: {'Enabled' if use_gradient_checkpointing else 'Disabled'}"
    )
    print(f"  Memory Monitoring: Target {memory_monitor.target_memory_gb}GB")
    print(
        f"  Enhanced Logging: WandB={'On' if use_wandb else 'Off'}, TensorBoard={'On' if use_tensorboard else 'Off'}"
    )
    print(
        f"  Advanced Thought Metrics: {'Enabled' if use_advanced_metrics else 'Disabled'}"
    )

    # Persistent thought stack configuration
    stack_reset_frequency = config.get("training", {}).get("stack_reset_frequency", 10)
    persistent_thought_stacks = None
    previous_new_thoughts = None  # For thought analyzer
    batch_count = 0
    print(f"  Thought Stack Persistence: Reset every {stack_reset_frequency} batches")

    # Check if training is already complete
    if start_epoch >= num_epochs:
        print(
            f"\n✅ Training already complete! Loaded model trained for {start_epoch} epochs."
        )
        print(f"Best evaluation loss: {best_eval_loss:.4f}")
        return

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        # Training phase
        model.train()
        epoch_losses = []

        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(train_pbar):
            # Move batch to device and ensure correct dtypes
            tokens = batch
            tokens = tokens.to(device, non_blocking=True)
            batch_size = tokens.shape[0]

            # Handle persistent thought stacks with reset frequency
            if (
                batch_count % stack_reset_frequency == 0
                or persistent_thought_stacks is None
                or persistent_thought_stacks.shape[0] != batch_size
            ):
                # Reset or initialize thought stacks
                thought_stacks = model.initialize_thought_stack(batch_size, device)
                # if batch_count % stack_reset_frequency == 0 and batch_count > 0:
                #     print(f"    🔄 Resetting thought stacks at batch {batch_count}")
            else:
                # Use persistent stacks from previous batch
                thought_stacks = persistent_thought_stacks

            # Compute loss with mixed precision
            if use_amp:
                with autocast("cuda"):
                    loss, metrics, evolved_stacks = compute_enhanced_loss_v2(
                        model,
                        (tokens, thought_stacks),
                        noise_scheduler,
                        device,
                        epoch,
                        config,
                        thought_analyzer,
                        previous_new_thoughts,
                    )
                    # Scale for gradient accumulation
                    loss = loss / gradient_accumulation_steps

                # Backward pass with scaler
                scaler.scale(loss).backward()
            else:
                # Standard precision
                loss, metrics, evolved_stacks = compute_enhanced_loss_v2(
                    model,
                    (tokens, thought_stacks),
                    noise_scheduler,
                    device,
                    epoch,
                    config,
                    thought_analyzer,
                    previous_new_thoughts,
                )
                # Scale for gradient accumulation
                loss = loss / gradient_accumulation_steps
                # Standard backward pass
                loss.backward()

            # Store evolved stacks and new thoughts for next batch (detached to prevent gradient accumulation)
            persistent_thought_stacks = evolved_stacks.detach()

            # Extract the new thoughts from the model's last forward pass for the analyzer
            # We'll get this from the metrics (it's already stored there)
            if "thought_norm" in metrics:
                # Create a dummy tensor for previous thoughts tracking
                # In a real implementation, we'd extract the actual new_thought from the model
                if previous_new_thoughts is None:
                    previous_new_thoughts = torch.zeros_like(
                        thought_stacks[:, -1]
                    )  # Use last stack entry as template
                else:
                    # For now, just use a placeholder - in practice you'd extract the actual new_thought
                    pass

            batch_count += 1

            # Optimizer step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_amp:
                    # Unscale gradients and clip
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    # Step with scaler
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard gradient clipping and step
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)  # More memory efficient
                lr_scheduler.step()

                # Add step-specific metrics
                metrics["grad_norm"] = grad_norm.item()
                metrics["learning_rate"] = lr_scheduler.get_last_lr()[0]
                metrics["optimizer_step"] = True
            else:
                # Fill in default values when not stepping
                metrics["grad_norm"] = 0.0
                metrics["learning_rate"] = lr_scheduler.get_last_lr()[0]
                metrics["optimizer_step"] = False

            epoch_losses.append(metrics)

            # Log metrics every few steps
            global_step = epoch * len(train_loader) + batch_idx
            if batch_idx % config.get("logging", {}).get("log_every", 50) == 0:
                enhanced_logger.log_metrics(metrics, global_step, "train")

            # Update progress bar
            stack_status = (
                "fresh" if batch_count % stack_reset_frequency == 1 else "persistent"
            )
            progress_info = {
                "loss": f"{metrics['total_loss']:.4f}",
                "ppl": f"{metrics['perplexity']:.1f}",
                "t_rms": f"{metrics['thought_rms']:.3f}",
                "t_sparse": f"{metrics['thought_sparsity']:.2f}",
                "lr": f"{lr_scheduler.get_last_lr()[0]:.2e}",
                "mem": f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB",
                "stack": stack_status,
            }

            # Add advanced metrics to progress bar if available
            if use_advanced_metrics and "advanced_thought_entropy" in metrics:
                progress_info["entropy"] = f"{metrics['advanced_thought_entropy']:.2f}"
                if "advanced_effective_dimensionality" in metrics:
                    progress_info["eff_dim"] = (
                        f"{metrics['advanced_effective_dimensionality']:.1f}"
                    )

            train_pbar.set_postfix(progress_info)

            # More aggressive memory monitoring and cleanup
            if batch_idx % 10 == 0:  # Check more frequently
                # Check memory usage
                memory_warning = memory_monitor.check_memory_usage(step=batch_idx)
                if memory_warning:
                    print(f"\n⚠️  Memory warning at batch {batch_idx}, cleaning up...")
                    # Aggressive cleanup if memory is high
                    memory_monitor.cleanup_memory()
                    # Force Python garbage collection
                    import gc

                    gc.collect()
                else:
                    # Regular cleanup every few batches
                    if batch_idx % 5 == 0:
                        torch.cuda.empty_cache()

        # Compute epoch averages
        epoch_metrics = {}
        for key in epoch_losses[0].keys():
            epoch_metrics[f"avg_{key}"] = np.mean([l[key] for l in epoch_losses])

        # Evaluation phase
        print("\n📊 Evaluating enhanced model...")
        model.eval()
        eval_losses = []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                tokens = batch
                tokens = tokens.to(device, non_blocking=True)

                # Initialize fresh thought stacks using model's method for each batch
                batch_size = tokens.shape[0]
                thought_stacks = model.initialize_thought_stack(batch_size, device)

                # Use autocast for evaluation too
                if use_amp:
                    with autocast("cuda"):
                        loss, metrics, _ = compute_enhanced_loss(
                            model,
                            (tokens, thought_stacks),
                            noise_scheduler,
                            device,
                            epoch,
                            config,
                        )
                else:
                    loss, metrics, _ = compute_enhanced_loss(
                        model,
                        (tokens, thought_stacks),
                        noise_scheduler,
                        device,
                        epoch,
                        config,
                    )
                eval_losses.append(metrics)

        # Average evaluation metrics
        eval_metrics = {}
        for key in eval_losses[0].keys():
            eval_metrics[key] = np.mean([l[key] for l in eval_losses])

        # Log evaluation metrics
        eval_step = (epoch + 1) * len(train_loader)
        enhanced_logger.log_metrics(eval_metrics, eval_step, "eval")

        # Log model analysis every epoch
        enhanced_logger.log_model_analysis(model, eval_step)

        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        memory_summary = memory_monitor.get_memory_summary()

        print(f"\n📈 Epoch {epoch + 1} Results:")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {epoch_metrics['avg_total_loss']:.4f}")
        print(f"    - Token Loss: {epoch_metrics['avg_token_loss']:.4f}")
        print(f"    - Perplexity: {epoch_metrics.get('avg_perplexity', 0):.1f}")
        print(f"    - Thought RMS: {epoch_metrics.get('avg_thought_rms', 0):.3f}")
        print(
            f"    - Thought Sparsity: {epoch_metrics.get('avg_thought_sparsity', 0):.2f}"
        )
        print(
            f"    - Thought Consistency: {epoch_metrics.get('avg_thought_consistency', 0):.2f}"
        )
        print(f"  Eval Loss: {eval_metrics['total_loss']:.4f}")
        print(f"    - Token Loss: {eval_metrics['token_loss']:.4f}")
        print(f"    - Perplexity: {eval_metrics.get('perplexity', 0):.1f}")
        print(f"    - Thought RMS: {eval_metrics.get('thought_rms', 0):.3f}")
        print(f"    - Thought Sparsity: {eval_metrics.get('thought_sparsity', 0):.2f}")
        print(f"  Learning Rate: {lr_scheduler.get_last_lr()[0]:.2e}")
        print(
            f"  Memory Usage: {memory_summary['current_memory_gb']:.1f}GB (Peak: {memory_summary['peak_memory_gb']:.1f}GB)"
        )
        if memory_summary["memory_warnings"] > 0:
            print(f"  Memory Warnings: {memory_summary['memory_warnings']}")

        # Save best model and check for early stopping
        if eval_metrics["total_loss"] < best_eval_loss:
            best_eval_loss = eval_metrics["total_loss"]
            no_improvement_count = 0  # Reset counter
            best_model_path = os.path.join(checkpoint_dir, f"best_enhanced_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 New best model saved! (eval_loss: {best_eval_loss:.4f})")
        else:
            no_improvement_count += 1
            print(
                f"⚠️  No improvement for {no_improvement_count}/{early_stopping_patience} epochs"
            )

            # Early stopping check
            if no_improvement_count >= early_stopping_patience:
                print(
                    f"🛑 Early stopping triggered! No improvement for {early_stopping_patience} epochs."
                )
                print(f"   Best eval loss: {best_eval_loss:.4f}")
                break

        # Save checkpoint using enhanced logger
        checkpoint_path = os.path.join(
            checkpoint_dir, f"enhanced_checkpoint_epoch_{epoch + 1}.pt"
        )
        checkpoint_metadata = {
            "eval_metrics": eval_metrics,
            "epoch_metrics": epoch_metrics,
            "param_count": param_count,
            "best_eval_loss": best_eval_loss,
            "memory_summary": memory_summary,
        }
        enhanced_logger.save_checkpoint_with_metadata(
            model,
            optimizer,
            lr_scheduler,
            epoch + 1,
            checkpoint_metadata,
            checkpoint_path,
        )
        print(f"💾 Enhanced checkpoint saved: {os.path.basename(checkpoint_path)}")

        # Aggressive memory cleanup between epochs
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc

        gc.collect()
        print(
            f"  Memory cleaned: {memory_monitor.get_current_memory_gb():.1f}GB current"
        )

    # Training complete
    total_training_time = time.time() - training_start_time

    print(f"\n🎉 Enhanced model training complete!")
    print(f"  Total time: {total_training_time / 60:.1f} minutes")
    print(f"  Best eval loss: {best_eval_loss:.4f}")
    print(f"  Final model size: {param_count / 1e6:.1f}M parameters")

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, f"final_enhanced_model.pt")
    torch.save(model.state_dict(), final_model_path)

    # Save training summary
    summary = {
        "config": config,
        "training_summary": {
            "dataset_name": args.dataset,
            "train_samples": args.train_samples,
            "eval_samples": args.eval_samples,
            "epochs_completed": num_epochs,
            "total_training_time_minutes": total_training_time / 60,
            "best_eval_loss": best_eval_loss,
            "final_eval_loss": eval_metrics["total_loss"],
            "parameter_count": param_count,
            "vocab_size": actual_vocab_size,
        },
        "final_metrics": eval_metrics,
    }

    summary_path = os.path.join(
        log_dir,
        f'enhanced_training_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Generate comprehensive training report
    training_report = enhanced_logger.create_training_report()
    print(f"📊 Training summary saved: {summary_path}")
    print(
        f"📊 Enhanced training report saved: {os.path.join(log_dir, 'training_report.json')}"
    )
    print(f"✅ Enhanced training completed successfully!")
    print(f"Best evaluation loss: {best_eval_loss:.4f}")

    # Close tensorboard writer if available
    if hasattr(enhanced_logger, "writer") and enhanced_logger.writer:
        try:
            enhanced_logger.writer.flush()
            enhanced_logger.writer.close()
            print("📊 TensorBoard writer closed successfully")
        except Exception as e:
            print(f"⚠️  Warning closing TensorBoard writer: {e}")


if __name__ == "__main__":
    main()
