#!/usr/bin/env python3
"""
Controlled Comparison Experiment: Thought Model vs Baseline Model
Train both models with identical conditions to test thought contribution

This script runs a scientific experiment to determine if the 3D thought system
actually contributes to model performance by comparing against an identical
baseline without thoughts.
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
import time
import math
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

# Import both models for comparison
from diffusion_thought_tensor.model.stacked_3d_model import (
    StackedDiffusionModel3D,
    MaskingStrategy,
)
from diffusion_thought_tensor.model.baseline_dream_model import BaselineDreamModel
from diffusion_thought_tensor.data.huggingface_data_loader import (
    create_huggingface_data_loaders,
)
from diffusion_thought_tensor.utils.training_utils import count_parameters
from diffusion_thought_tensor.utils.loss_utils import compute_dream_enhanced_loss
from diffusion_thought_tensor.utils.noise_schedules import create_noise_scheduler
from dream_metrics_analyzer_lite import ThoughtAnalyzer, compare_dream_effectiveness


class ComparisonExperiment:
    """Run controlled comparison between thought and baseline models"""

    def __init__(self, args):
        self.args = args

        # Set random seeds for reproducibility
        self.set_seeds(args.seed)

        # Load configs
        self.thought_config = self._load_config(args.thought_config)
        self.baseline_config = self._load_config(args.baseline_config)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔬 Using device: {self.device}")

        # Create models
        self.thought_model = self._create_thought_model()
        self.baseline_model = self._create_baseline_model()

        # Setup data loaders
        self.train_loader, self.eval_loader = self._setup_data()

        # Setup optimizers and schedulers
        self.thought_optimizer = self._create_optimizer(
            self.thought_model, self.thought_config
        )
        self.baseline_optimizer = self._create_optimizer(
            self.baseline_model, self.baseline_config
        )

        # Create noise scheduler for diffusion-based training
        self.noise_scheduler = create_noise_scheduler(self.thought_config.get("diffusion", {
            "num_steps": 1000,
            "schedule_type": "cosine",
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "prediction_type": "epsilon"
        }))
        self.pad_token_id = 50256  # GPT2's eos_token_id used as pad_token
        
        # Stack reset frequency configuration
        self.stack_reset_frequency = self.thought_config.get("training", {}).get("stack_reset_frequency", 10)
        
        # Setup dual learning rate schedulers (cosine + plateau)
        # For thought model
        warmup_steps = self.thought_config["training"]["warmup_steps"]
        self.thought_cosine_scheduler = CosineAnnealingWarmRestarts(
            self.thought_optimizer,
            T_0=warmup_steps,
            T_mult=2,
            eta_min=float(self.thought_config["training"]["learning_rate"]) * 0.01,
        )
        self.thought_plateau_scheduler = ReduceLROnPlateau(
            self.thought_optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            min_lr=float(self.thought_config["training"]["learning_rate"]) * 0.001
        )
        
        # For baseline model
        self.baseline_cosine_scheduler = CosineAnnealingWarmRestarts(
            self.baseline_optimizer,
            T_0=warmup_steps,
            T_mult=2,
            eta_min=float(self.baseline_config["training"]["learning_rate"]) * 0.01,
        )
        self.baseline_plateau_scheduler = ReduceLROnPlateau(
            self.baseline_optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            min_lr=float(self.baseline_config["training"]["learning_rate"]) * 0.001
        )

        # Setup mixed precision with much lower initial scale for stability
        self.thought_scaler = (
            GradScaler(init_scale=2**8)  # 256 instead of 1024
            if self.thought_config["hardware"]["mixed_precision"]
            else None
        )
        self.baseline_scaler = (
            GradScaler(init_scale=2**8)  # 256 instead of 1024
            if self.baseline_config["hardware"]["mixed_precision"]
            else None
        )

        # Setup logging
        self.setup_logging()

        # Initialize thought analyzer for comprehensive metrics
        self.thought_analyzer = ThoughtAnalyzer()

    def set_seeds(self, seed):
        """Set all random seeds for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"🎲 Set random seed to {seed}")

    def _load_config(self, config_path):
        """Load configuration file"""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _create_thought_model(self):
        """Create thought-enabled model"""
        config = self.thought_config

        masking_strategy = MaskingStrategy(
            mask_ratio=config["masking"]["mask_ratio"],
            confidence_threshold=config["masking"]["confidence_threshold"],
            progressive_unmasking=config["masking"]["progressive_unmasking"],
            block_size=config["masking"]["block_size"],
        )

        # Extract thought tensor config
        thought_config = config["thought_tensor"]
        input_dims = thought_config["input_dims"]  # [H, W, channels]

        model = StackedDiffusionModel3D(
            vocab_size=config["model"]["vocab_size"],
            embed_dim=config["model"]["embed_dim"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            thought_dims=tuple(input_dims[:2]),  # H, W
            stack_depth=thought_config["stack_size"],
            thought_hidden_dim=input_dims[2],  # channels
            max_seq_length=config["model"]["max_seq_length"],
            dropout=config["model"]["dropout"],
            masking_strategy=masking_strategy,
        ).to(self.device)

        print(f"✅ Thought model: {count_parameters(model) / 1e6:.1f}M parameters")
        return model

    def _create_baseline_model(self):
        """Create baseline model (no thoughts)"""
        config = self.baseline_config

        masking_strategy = MaskingStrategy(
            mask_ratio=config["masking"]["mask_ratio"],
            confidence_threshold=config["masking"]["confidence_threshold"],
            progressive_unmasking=config["masking"]["progressive_unmasking"],
            block_size=config["masking"]["block_size"],
        )

        model = BaselineDreamModel(
            vocab_size=config["model"]["vocab_size"],
            embed_dim=config["model"]["embed_dim"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            max_seq_length=config["model"]["max_seq_length"],
            dropout=config["model"]["dropout"],
            masking_strategy=masking_strategy,
        ).to(self.device)

        print(f"✅ Baseline model: {count_parameters(model) / 1e6:.1f}M parameters")
        return model

    def _setup_data(self):
        """Setup data loaders with identical data for fair comparison"""
        print(f"📊 Loading dataset: {self.args.dataset}")

        # Create a config matching the expected structure
        data_config = {
            "model": {"max_seq_length": self.args.seq_length},
            "training": {"batch_size": self.thought_config["training"]["batch_size"]},
            "hardware": {"num_workers": 0, "dataloader_pin_memory": False},
            "data_cache_dir": "data/cache",
        }

        train_loader, eval_loader = create_huggingface_data_loaders(
            config=data_config,
            dataset_name=self.args.dataset,
            train_samples=self.args.train_samples,
            eval_samples=self.args.eval_samples,
        )

        print(
            f"📊 Data loaded: {len(train_loader)} train batches, {len(eval_loader)} eval batches"
        )
        return train_loader, eval_loader

    def _create_optimizer(self, model, config):
        """Create optimizer with identical settings"""
        opt_config = config["optimizer"]
        lr = float(config["training"]["learning_rate"])  # Ensure it's a float
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=float(opt_config["weight_decay"]),
            betas=[float(b) for b in opt_config["betas"]],
            eps=float(opt_config["eps"]),
        )


    def setup_logging(self):
        """Setup logging directories and files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create comparison experiment directory
        self.experiment_dir = f"outputs/comparison_experiment_{timestamp}"
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Create subdirectories for each model
        self.thought_log_dir = os.path.join(self.experiment_dir, "thought_model")
        self.baseline_log_dir = os.path.join(self.experiment_dir, "baseline_model")
        os.makedirs(self.thought_log_dir, exist_ok=True)
        os.makedirs(self.baseline_log_dir, exist_ok=True)

        # Save experiment configuration
        experiment_info = {
            "experiment_type": "thought_vs_baseline_comparison",
            "timestamp": timestamp,
            "args": vars(self.args),
            "thought_params": count_parameters(self.thought_model),
            "baseline_params": count_parameters(self.baseline_model),
            "dataset": self.args.dataset,
            "train_samples": self.args.train_samples,
            "eval_samples": self.args.eval_samples,
        }

        with open(os.path.join(self.experiment_dir, "experiment_info.json"), "w") as f:
            json.dump(experiment_info, f, indent=2)

        print(f"📝 Logging to: {self.experiment_dir}")

    def train_epoch(self, model, optimizer, cosine_scheduler, scaler, model_name, epoch, mask_ratio, confidence_threshold):
        """Train one epoch for a model using DREAM-enhanced loss"""
        model.train()
        epoch_losses = []
        
        # Get gradient accumulation steps from config
        config = (
            self.thought_config if model_name == "Thought" else self.baseline_config
        )
        gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"]
        
        # Initialize persistent thought stack for the epoch (if thought model)
        persistent_thought_stack = None
        batch_count = 0
        reset_stack_this_epoch = False
        
        # Reset stack flag for this epoch if frequency is 0
        if self.stack_reset_frequency == 0:
            reset_stack_this_epoch = True
            
        if model_name == "Thought":
            # Start with a reasonable batch size for initialization
            init_batch_size = self.thought_config["training"]["batch_size"]
            persistent_thought_stack = model.initialize_stack(
                init_batch_size, self.device
            )

        pbar = tqdm(self.train_loader, desc=f"{model_name} Epoch {epoch}")
        accumulation_step = 0
        
        for batch_idx, batch in enumerate(pbar):
            # Handle different batch formats
            if isinstance(batch, dict):
                tokens = batch["input_ids"].to(self.device)
            else:
                tokens = batch.to(self.device)

            # Ensure tokens is 2D
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)

            batch_size = tokens.shape[0]
            
            # Handle persistent thought stacks with reset strategy
            reset_stack = False
            
            if model_name == "Thought":
                if self.stack_reset_frequency == 0:
                    # Reset once per epoch when frequency is 0
                    if reset_stack_this_epoch or persistent_thought_stack is None:
                        reset_stack = True
                        reset_stack_this_epoch = False  # Only reset once per epoch
                else:
                    # Normal frequency-based reset
                    if batch_count % self.stack_reset_frequency == 0:
                        reset_stack = True
                        
                # Always reset if batch size changes or no existing stacks
                if (
                    persistent_thought_stack is None
                    or persistent_thought_stack.shape[0] != batch_size
                ):
                    reset_stack = True
                    
                if reset_stack:
                    current_thought_stack = model.initialize_stack(batch_size, self.device)
                else:
                    current_thought_stack = persistent_thought_stack
            else:
                # Baseline model doesn't use thought stacks
                current_thought_stack = None

            # Only zero gradients at the start of accumulation cycle
            if accumulation_step == 0:
                optimizer.zero_grad()

            # Compute loss with DREAM-enhanced loss function
            if scaler is not None:
                with autocast(device_type="cuda"):
                    if model_name == "Thought":
                        loss, metrics, evolved_stacks = compute_dream_enhanced_loss(
                            model, tokens, current_thought_stack, self.noise_scheduler,
                            self.device, epoch, config, mask_ratio, confidence_threshold, self.pad_token_id
                        )
                        persistent_thought_stack = evolved_stacks.detach()
                    else:
                        # For baseline, create dummy stack (won't be used but needed for function signature)
                        dummy_stack = torch.zeros(batch_size, 64, 16, 16, 8, device=self.device)
                        loss, metrics, _ = compute_dream_enhanced_loss(
                            model, tokens, dummy_stack, self.noise_scheduler,
                            self.device, epoch, config, mask_ratio, confidence_threshold, self.pad_token_id
                        )
                    
                    # Scale loss by accumulation steps
                    loss = loss / gradient_accumulation_steps

                scaler.scale(loss).backward()

                # Only step optimizer after accumulating gradients
                if (accumulation_step + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)

                    # Monitor gradient norms before clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float("inf")
                    )
                    
                    # Skip batch if gradients are extremely large (anomaly)
                    if grad_norm > 100.0:
                        print(f"\n🚫 Skipping batch - extreme gradient: {grad_norm:.2f}")
                        optimizer.zero_grad()  # Clear accumulated gradients
                        continue
                    
                    if grad_norm > 10.0:  # Log large gradients
                        print(
                            f"\n⚠️ Large gradient detected: {grad_norm:.2f} (will clip to {config['training']['max_grad_norm']})"
                        )

                    # Apply actual clipping
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["training"]["max_grad_norm"]
                    )
                    scaler.step(optimizer)
                    scaler.update()

                    # Step cosine scheduler after optimizer
                    cosine_scheduler.step()
                    
                    metrics["grad_norm"] = grad_norm.item()
                    metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
            else:
                # Without mixed precision
                if model_name == "Thought":
                    loss, metrics, evolved_stacks = compute_dream_enhanced_loss(
                        model, tokens, current_thought_stack, self.noise_scheduler,
                        self.device, epoch, config, mask_ratio, confidence_threshold, self.pad_token_id
                    )
                    persistent_thought_stack = evolved_stacks.detach()
                else:
                    dummy_stack = torch.zeros(batch_size, 64, 16, 16, 8, device=self.device)
                    loss, metrics, _ = compute_dream_enhanced_loss(
                        model, tokens, dummy_stack, self.noise_scheduler,
                        self.device, epoch, config, mask_ratio, confidence_threshold, self.pad_token_id
                    )
                
                # Scale loss by accumulation steps
                loss = loss / gradient_accumulation_steps

                loss.backward()

                # Only step optimizer after accumulating gradients
                if (accumulation_step + 1) % gradient_accumulation_steps == 0:
                    # Monitor gradient norms before clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float("inf")
                    )
                    
                    # Skip batch if gradients are extremely large (anomaly)
                    if grad_norm > 100.0:
                        print(f"\n🚫 Skipping batch - extreme gradient: {grad_norm:.2f}")
                        optimizer.zero_grad()  # Clear accumulated gradients
                        continue
                        
                    if grad_norm > 10.0:  # Log large gradients
                        print(
                            f"\n⚠️ Large gradient detected: {grad_norm:.2f} (will clip to {config['training']['max_grad_norm']})"
                        )

                    # Apply actual clipping
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config["training"]["max_grad_norm"]
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Step cosine scheduler
                    cosine_scheduler.step()
                    
                    metrics["grad_norm"] = grad_norm.item()
                    metrics["learning_rate"] = optimizer.param_groups[0]["lr"]

            epoch_losses.append(metrics)
            
            if model_name == "Thought":
                batch_count += 1
            
            accumulation_step += 1
            
            # Update progress bar with DREAM metrics
            progress_info = {
                "loss": f"{metrics['total_loss']:.4f}",
                "tok_loss": f"{metrics['token_loss']:.4f}",
                "m_acc": f"{metrics['masked_accuracy']:.3f}",
                "u_acc": f"{metrics['unmasked_accuracy']:.3f}",
                "t_rms": f"{metrics['thought_rms']:.3f}",
                "mask": f"{metrics['mask_ratio_actual']:.2f}",
            }
            
            if "learning_rate" in metrics:
                progress_info["lr"] = f"{metrics['learning_rate']:.2e}"
                
            # Add DREAM-specific metrics to progress
            if "bidirectional_effectiveness" in metrics:
                progress_info["bi_eff"] = f"{metrics['bidirectional_effectiveness']:.2f}"
                
            pbar.set_postfix(progress_info)
            
            # Memory cleanup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
        
        # Compute epoch averages
        epoch_metrics = {}
        for key in epoch_losses[0].keys():
            values = []
            for l in epoch_losses:
                if key not in l:
                    continue
                val = l[key]
                if torch.is_tensor(val):
                    val = float(val.detach().cpu().item()) if val.numel() == 1 else val.detach().cpu().numpy()
                elif hasattr(val, 'item'):
                    val = float(val.item())
                values.append(val)
            if values:
                epoch_metrics[f"avg_{key}"] = np.mean(values)
                
        return epoch_metrics["avg_total_loss"] if "avg_total_loss" in epoch_metrics else 999.0

    def evaluate_model(self, model, model_name, mask_ratio, confidence_threshold):
        """Evaluate a model using DREAM-enhanced loss"""
        model.eval()
        eval_losses = []
        
        # Get config
        config = (
            self.thought_config if model_name == "Thought" else self.baseline_config
        )
        
        # Initialize persistent thought stack for evaluation (if thought model)
        persistent_thought_stack = None
        if model_name == "Thought":
            init_batch_size = self.thought_config["training"]["batch_size"]
            persistent_thought_stack = model.initialize_stack(
                init_batch_size, self.device
            )
            # Reset thought analyzer for this evaluation
            self.thought_analyzer.reset()

        # Initialize comprehensive metrics storage for thought analyzer
        comprehensive_metrics = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.eval_loader, desc=f"Eval {model_name}")):
                # Handle different batch formats
                if isinstance(batch, dict):
                    tokens = batch["input_ids"].to(self.device)
                else:
                    tokens = batch.to(self.device)

                # Ensure tokens is 2D
                if tokens.dim() == 1:
                    tokens = tokens.unsqueeze(0)

                batch_size = tokens.shape[0]

                # Initialize or reuse evaluation thought stacks
                if model_name == "Thought":
                    if (
                        persistent_thought_stack is None
                        or persistent_thought_stack.shape[0] != batch_size
                    ):
                        current_thought_stack = model.initialize_stack(batch_size, self.device)
                    else:
                        current_thought_stack = persistent_thought_stack
                        
                    # Use DREAM-enhanced loss for evaluation
                    loss, metrics, evolved_stacks = compute_dream_enhanced_loss(
                        model, tokens, current_thought_stack, self.noise_scheduler,
                        self.device, 0, config, mask_ratio, confidence_threshold, self.pad_token_id
                    )
                    
                    # Skip comprehensive thought metrics - already handled by DREAM loss function
                    # The compute_dream_enhanced_loss already provides comprehensive metrics
                        
                    # Update persistent stack for next batch
                    persistent_thought_stack = evolved_stacks.detach()
                else:
                    # For baseline, create dummy stack
                    dummy_stack = torch.zeros(batch_size, 64, 16, 16, 8, device=self.device)
                    loss, metrics, _ = compute_dream_enhanced_loss(
                        model, tokens, dummy_stack, self.noise_scheduler,
                        self.device, 0, config, mask_ratio, confidence_threshold, self.pad_token_id
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
            
        # Rename keys for backward compatibility
        result = {
            "loss": eval_metrics.get("total_loss", 999.0),
            "accuracy": eval_metrics.get("masked_accuracy", 0.0),  # Use masked accuracy as primary metric
            "perplexity": math.exp(eval_metrics.get("total_loss", 999.0)),
            "masked_accuracy": eval_metrics.get("masked_accuracy", 0.0),
            "unmasked_accuracy": eval_metrics.get("unmasked_accuracy", 0.0),
            "bidirectional_effectiveness": eval_metrics.get("bidirectional_effectiveness", 1.0),
            "masking_gap": eval_metrics.get("unmasked_accuracy", 0.0) - eval_metrics.get("masked_accuracy", 0.0),
            "thought_rms": eval_metrics.get("thought_rms", 0.0),
            "masked_tokens": eval_metrics.get("masked_tokens", 0),
            "unmasked_tokens": eval_metrics.get("unmasked_tokens", 0),
        }

        # Comprehensive thought metrics are now provided directly by compute_dream_enhanced_loss
        # in the eval_metrics dictionary, so no additional processing needed

        return result

    def run_experiment(self):
        """Run the complete comparison experiment"""
        print("\n" + "=" * 80)
        print("🔬 STARTING CONTROLLED COMPARISON EXPERIMENT")
        print("=" * 80)
        print(
            f"Thought Model: {count_parameters(self.thought_model) / 1e6:.1f}M parameters"
        )
        print(
            f"Baseline Model: {count_parameters(self.baseline_model) / 1e6:.1f}M parameters"
        )
        print(f"Dataset: {self.args.dataset}")
        print(f"Epochs: {self.args.epochs}")
        print(f"Random Seed: {self.args.seed}")
        
        # Progressive masking and confidence scheduling
        initial_mask_ratio = self.thought_config["masking"]["mask_ratio"]
        final_mask_ratio = initial_mask_ratio * 0.5
        initial_confidence = self.thought_config["masking"]["confidence_threshold"]
        final_confidence = min(0.9, initial_confidence + 0.15)
        
        print(f"Progressive masking: {initial_mask_ratio:.2f} → {final_mask_ratio:.2f}")
        print(f"Progressive confidence: {initial_confidence:.2f} → {final_confidence:.2f}")

        # Store results for comparison
        results = {
            "thought_model": {"train_losses": [], "eval_metrics": []},
            "baseline_model": {"train_losses": [], "eval_metrics": []},
        }

        for epoch in range(1, self.args.epochs + 1):
            print(f"\n📅 Epoch {epoch}/{self.args.epochs}")
            
            # Calculate progressive parameters
            progress = (epoch - 1) / max(1, self.args.epochs - 1)
            current_mask_ratio = initial_mask_ratio + (final_mask_ratio - initial_mask_ratio) * progress
            current_confidence = initial_confidence + (final_confidence - initial_confidence) * progress
            
            print(f"Using mask ratio: {current_mask_ratio:.3f}, confidence: {current_confidence:.3f}")

            # Train both models
            thought_loss = self.train_epoch(
                self.thought_model,
                self.thought_optimizer,
                self.thought_cosine_scheduler,
                self.thought_scaler,
                "Thought",
                epoch,
                current_mask_ratio,
                current_confidence,
            )

            baseline_loss = self.train_epoch(
                self.baseline_model,
                self.baseline_optimizer,
                self.baseline_cosine_scheduler,
                self.baseline_scaler,
                "Baseline",
                epoch,
                current_mask_ratio,
                current_confidence,
            )

            # Evaluate both models
            thought_metrics = self.evaluate_model(self.thought_model, "Thought", current_mask_ratio, current_confidence)
            baseline_metrics = self.evaluate_model(self.baseline_model, "Baseline", current_mask_ratio, current_confidence)
            
            # Update plateau schedulers after evaluation
            self.thought_plateau_scheduler.step(thought_metrics["loss"])
            self.baseline_plateau_scheduler.step(baseline_metrics["loss"])

            # Store results
            results["thought_model"]["train_losses"].append(thought_loss)
            results["thought_model"]["eval_metrics"].append(thought_metrics)
            results["baseline_model"]["train_losses"].append(baseline_loss)
            results["baseline_model"]["eval_metrics"].append(baseline_metrics)
            
            # Memory cleanup
            torch.cuda.empty_cache()

            # Print comparison
            print(f"\n📊 Epoch {epoch} Results:")
            print(
                f"{'Model':<15} {'Train Loss':<12} {'Eval Loss':<12} {'Accuracy':<10} {'Perplexity':<12}"
            )
            print("-" * 65)
            print(
                f"{'Thought':<15} {thought_loss:<12.4f} {thought_metrics['loss']:<12.4f} "
                f"{thought_metrics['accuracy']:<10.4f} {thought_metrics['perplexity']:<12.2f}"
            )
            print(
                f"{'Baseline':<15} {baseline_loss:<12.4f} {baseline_metrics['loss']:<12.4f} "
                f"{baseline_metrics['accuracy']:<10.4f} {baseline_metrics['perplexity']:<12.2f}"
            )

            # Calculate differences
            loss_diff = baseline_metrics["loss"] - thought_metrics["loss"]
            acc_diff = thought_metrics["accuracy"] - baseline_metrics["accuracy"]
            ppl_diff = baseline_metrics["perplexity"] - thought_metrics["perplexity"]

            print(
                f"{'Difference':<15} {'':<12} {loss_diff:<12.4f} "
                f"{acc_diff:<10.4f} {ppl_diff:<12.2f}"
            )

            # Print DREAM-specific metrics if available
            if (
                "masked_accuracy" in thought_metrics
                and "masked_accuracy" in baseline_metrics
            ):
                print(f"\n🎭 DREAM Metrics:")
                print(
                    f"{'Model':<15} {'Masked Acc':<12} {'Unmasked Acc':<12} {'Bidir Eff':<12} {'Mask Gap':<12}"
                )
                print("-" * 67)

                t_masked = thought_metrics.get("masked_accuracy", 0)
                t_unmasked = thought_metrics.get("unmasked_accuracy", 0)
                t_bidir = thought_metrics.get("bidirectional_effectiveness", 1)
                t_gap = thought_metrics.get("masking_gap", 0)

                b_masked = baseline_metrics.get("masked_accuracy", 0)
                b_unmasked = baseline_metrics.get("unmasked_accuracy", 0)
                b_bidir = baseline_metrics.get("bidirectional_effectiveness", 1)
                b_gap = baseline_metrics.get("masking_gap", 0)

                print(
                    f"{'Thought':<15} {t_masked:<12.4f} {t_unmasked:<12.4f} {t_bidir:<12.4f} {t_gap:<12.4f}"
                )
                print(
                    f"{'Baseline':<15} {b_masked:<12.4f} {b_unmasked:<12.4f} {b_bidir:<12.4f} {b_gap:<12.4f}"
                )

                # DREAM-specific differences
                masked_diff = t_masked - b_masked
                unmasked_diff = t_unmasked - b_unmasked
                bidir_diff = t_bidir - b_bidir
                gap_diff = t_gap - b_gap

                print(
                    f"{'Difference':<15} {masked_diff:<12.4f} {unmasked_diff:<12.4f} {bidir_diff:<12.4f} {gap_diff:<12.4f}"
                )

                if masked_diff > 0.02:  # 2% improvement threshold for masked tokens
                    print(
                        "✅ Thoughts significantly improve masked token prediction (DREAM core benefit)"
                    )
                elif masked_diff > unmasked_diff:
                    print(
                        "🟡 Thoughts help more with masked tokens than unmasked (moderate DREAM benefit)"
                    )
                else:
                    print("❌ No clear DREAM-specific benefit from thoughts")

            if loss_diff > 0:
                print("✅ Overall: Thought model is performing better")
            else:
                print("❌ Overall: Baseline model is performing better")

            # Use DREAM-specific comparison analysis
            if (
                "masked_accuracy" in thought_metrics
                and "masked_accuracy" in baseline_metrics
            ):
                dream_comparison = compare_dream_effectiveness(
                    thought_metrics, baseline_metrics
                )
                if dream_comparison:
                    print(
                        f"\n🔍 DREAM Analysis: {dream_comparison.get('interpretation', 'Analysis unavailable')}"
                    )
                    print(
                        f"📋 Recommendation: {dream_comparison.get('recommendation', 'No specific recommendation')}"
                    )

                    # Log detailed DREAM comparison metrics
                    for key, value in dream_comparison.items():
                        if key not in [
                            "interpretation",
                            "recommendation",
                        ] and isinstance(value, (int, float)):
                            print(f"   {key}: {value:+.4f}")

        # Save final results
        final_results_path = os.path.join(self.experiment_dir, "final_results.json")
        with open(final_results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Print final analysis
        self.analyze_final_results(results)

        return results

    def analyze_final_results(self, results):
        """Analyze and interpret final results"""
        print("\n" + "=" * 80)
        print("🎯 FINAL ANALYSIS")
        print("=" * 80)

        # Get final metrics
        final_thought = results["thought_model"]["eval_metrics"][-1]
        final_baseline = results["baseline_model"]["eval_metrics"][-1]

        # Calculate improvements
        loss_improvement = (
            (final_baseline["loss"] - final_thought["loss"])
            / final_baseline["loss"]
            * 100
        )
        acc_improvement = (
            (final_thought["accuracy"] - final_baseline["accuracy"])
            / final_baseline["accuracy"]
            * 100
        )
        ppl_improvement = (
            (final_baseline["perplexity"] - final_thought["perplexity"])
            / final_baseline["perplexity"]
            * 100
        )

        print(f"Final Thought Model Performance:")
        print(f"  Loss: {final_thought['loss']:.4f}")
        print(f"  Accuracy: {final_thought['accuracy']:.4f}")
        print(f"  Perplexity: {final_thought['perplexity']:.2f}")

        print(f"\nFinal Baseline Model Performance:")
        print(f"  Loss: {final_baseline['loss']:.4f}")
        print(f"  Accuracy: {final_baseline['accuracy']:.4f}")
        print(f"  Perplexity: {final_baseline['perplexity']:.2f}")

        print(f"\nThought Model Improvements:")
        print(f"  Loss improvement: {loss_improvement:+.2f}%")
        print(f"  Accuracy improvement: {acc_improvement:+.2f}%")
        print(f"  Perplexity improvement: {ppl_improvement:+.2f}%")

        # Interpretation
        print(f"\n🧠 INTERPRETATION:")
        if loss_improvement > 5:
            print(
                "✅ STRONG EVIDENCE: Thoughts significantly contribute to performance"
            )
        elif loss_improvement > 2:
            print("🟡 MODERATE EVIDENCE: Thoughts provide modest improvements")
        elif loss_improvement > -2:
            print("⚖️ INCONCLUSIVE: No significant difference")
        else:
            print("❌ THOUGHTS ARE COUNTERPRODUCTIVE: Baseline performs better")

        # Parameter efficiency
        thought_params = count_parameters(self.thought_model)
        baseline_params = count_parameters(self.baseline_model)
        param_overhead = (thought_params - baseline_params) / baseline_params * 100

        print(f"\n📊 PARAMETER EFFICIENCY:")
        print(f"  Thought model: {thought_params / 1e6:.1f}M parameters")
        print(f"  Baseline model: {baseline_params / 1e6:.1f}M parameters")
        print(f"  Overhead: {param_overhead:.1f}%")

        if loss_improvement > 0:
            efficiency = loss_improvement / param_overhead
            print(f"  Efficiency: {efficiency:.3f}% improvement per 1% param increase")


def main():
    parser = argparse.ArgumentParser(description="Run controlled comparison experiment")

    parser.add_argument(
        "--thought_config",
        default="diffusion_thought_tensor/configs/dream_config.yaml",
        help="Config for thought model",
    )
    parser.add_argument(
        "--baseline_config",
        default="diffusion_thought_tensor/configs/baseline_config.yaml",
        help="Config for baseline model",
    )
    parser.add_argument(
        "--dataset", default="roneneldan/TinyStories", help="Dataset to use"
    )
    parser.add_argument(
        "--train_samples", type=int, default=10000, help="Number of training samples"
    )
    parser.add_argument(
        "--eval_samples", type=int, default=1000, help="Number of evaluation samples"
    )
    parser.add_argument("--seq_length", type=int, default=1024, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Run experiment
    experiment = ComparisonExperiment(args)
    results = experiment.run_experiment()

    print(f"\n✅ Experiment complete! Results saved to: {experiment.experiment_dir}")


if __name__ == "__main__":
    main()
