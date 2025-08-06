"""
Logging utilities for DREAM-Enhanced Model Training
"""

import torch
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, Optional


class DreamTrainingLogger:
    """Enhanced training logger for DREAM model with comprehensive metrics"""

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
                    project="dream-diffusion-thought-tensor",
                    config=config,
                    name=f"dream_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
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
                print("⚠️  Warning: tensorboard not available")
                self.use_tensorboard = False

        self.metrics_history = []
        self.best_metrics = {}

    def log_metrics(self, metrics, step, phase="train", model=None):
        """Enhanced metrics logging with DREAM-specific categories"""
        # Convert tensor values to floats for JSON serialization
        serializable_metrics = {}
        for k, v in metrics.items():
            if torch.is_tensor(v):
                if v.numel() == 1:
                    serializable_metrics[k] = float(v.item())
                else:
                    serializable_metrics[k] = v.detach().cpu().numpy().tolist()
            elif isinstance(v, np.ndarray):
                serializable_metrics[k] = v.tolist()
            elif isinstance(v, (int, float, str, bool, list, dict, type(None))):
                serializable_metrics[k] = v
            elif hasattr(v, 'item'):  # Handle numpy scalars
                serializable_metrics[k] = float(v.item())
            else:
                # Convert other types to string as fallback
                serializable_metrics[k] = str(v)
        
        metrics_with_meta = {
            "step": step,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            **serializable_metrics,
        }

        self.metrics_history.append(metrics_with_meta)

        # Log to wandb
        if self.use_wandb:
            try:
                import wandb
                wandb.log({f"{phase}/{k}": v for k, v in metrics.items()}, step=step)
            except ImportError:
                pass

        # Enhanced TensorBoard logging with categories
        if self.use_tensorboard and hasattr(self, "writer"):
            try:
                # Categorize metrics for better organization
                categories = {
                    "Losses": {},
                    "Accuracy": {},
                    "DREAM": {},
                    "Thoughts": {},
                    "Masking": {},
                    "Training": {},
                }
                
                for key, value in metrics.items():
                    if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                        continue
                    
                    # Categorize metrics
                    if "loss" in key:
                        categories["Losses"][key] = value
                    elif "accuracy" in key or "perplexity" in key:
                        categories["Accuracy"][key] = value
                    elif any(x in key for x in ["bidirectional", "adaptive", "confidence", "block", "mask_ratio"]):
                        categories["DREAM"][key] = value
                    elif "thought" in key:
                        categories["Thoughts"][key] = value
                    elif "mask" in key:
                        categories["Masking"][key] = value
                    else:
                        categories["Training"][key] = value
                
                # Log categorized metrics
                for category, cat_metrics in categories.items():
                    for key, value in cat_metrics.items():
                        self.writer.add_scalar(f"{phase}/{category}/{key}", value, step)
                
                # Add histograms for thought activations if available
                if phase == "train" and step % 200 == 0 and model is not None:
                    try:
                        if hasattr(model, '_last_thought_stats'):
                            stats = model._last_thought_stats
                            if 'thought_activations' in stats:
                                self.writer.add_histogram(
                                    f"{phase}/Distributions/thought_activations", 
                                    stats['thought_activations'], 
                                    step
                                )
                    except Exception:
                        pass
                
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