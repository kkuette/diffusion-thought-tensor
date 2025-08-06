"""
Loss computation utilities for DREAM-Enhanced Model Training
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from ..model.stacked_3d_model import StackedDiffusionModel3D


def compute_dream_enhanced_loss(
    model: StackedDiffusionModel3D,
    batch: torch.Tensor,
    thought_stacks: torch.Tensor,
    noise_scheduler,
    device: torch.device,
    epoch: int = 0,
    config: Optional[Dict] = None,
    masking_ratio: float = 0.7,
    confidence_threshold: float = 0.75,
    pad_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    """
    Comprehensive loss function for DREAM-enhanced model
    Includes all DREAM improvements and self-supervised thought learning
    """
    
    tokens = batch
    batch_size, seq_length = tokens.shape

    # Sample timesteps for diffusion
    timesteps = torch.randint(
        0, noise_scheduler.num_steps, (batch_size,), device=device
    )

    # Create masking pattern with DREAM-style strategy
    token_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)

    for b in range(batch_size):
        if pad_token_id is not None:
            non_pad_positions = (tokens[b] != pad_token_id).nonzero(as_tuple=True)[0]
            if len(non_pad_positions) > 0:
                non_pad_count = len(non_pad_positions)
                mask_count = int(non_pad_count * masking_ratio)
                if mask_count > 0:
                    selected_indices = torch.randperm(non_pad_count, device=device)[:mask_count]
                    mask_positions = non_pad_positions[selected_indices]
                    token_mask[b, mask_positions] = True
        else:
            mask_count = int(seq_length * masking_ratio)
            mask_positions = torch.randperm(seq_length, device=device)[:mask_count]
            token_mask[b, mask_positions] = True

    # Detect model type and call appropriately
    if hasattr(model, 'thought_stack'):
        # StackedDiffusionModel3D (thought model)
        forward_outputs = model(tokens, thought_stacks, timesteps, token_mask, 
                               return_all_thoughts=False)
        pred_logits, new_thought, _ = forward_outputs
        thought_losses = {}  # 3D model doesn't use self-supervised thought losses
        
        # Evolve thought stack using 3D model's push method
        evolved_thought_stacks = model.thought_stack.push_with_importance(
            thought_stacks, new_thought
        )
    else:
        # BaselineDreamModel (no thoughts)
        pred_logits = model(tokens, timesteps, token_mask)
        new_thought = torch.zeros_like(thought_stacks[:, :, :, :, 0])  # Dummy thought
        thought_losses = {}
        evolved_thought_stacks = thought_stacks  # No evolution for baseline

    # === LOSS COMPONENTS ===
    
    # 1. Primary masked token prediction loss
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

    # 2. Bidirectional confidence loss (DREAM-style)
    unmasked_logits = pred_logits[~token_mask]
    if unmasked_logits.numel() > 0:
        unmasked_probs = F.softmax(unmasked_logits, dim=-1)
        unmasked_entropy = -torch.sum(
            unmasked_probs * torch.log(unmasked_probs + 1e-8), dim=-1
        )
        confidence_loss = unmasked_entropy.mean()
    else:
        confidence_loss = torch.tensor(0.0, device=device)

    # 3. Self-supervised thought losses
    thought_loss_total = torch.tensor(0.0, device=device)
    if thought_losses:
        for loss_name, loss_value in thought_losses.items():
            thought_loss_total += loss_value

    # 4. Thought consistency loss
    thought_consistency_loss = torch.tensor(0.0, device=device)
    if epoch > 2:
        thought_norm = torch.norm(new_thought, dim=-1).mean()
        thought_consistency_loss = 0.01 * thought_norm

    # Combine losses with curriculum learning
    curriculum_weight = min(1.0, epoch / 10.0)
    total_loss = (
        token_loss + 
        0.1 * confidence_loss + 
        0.05 * curriculum_weight * thought_loss_total +
        0.01 * thought_consistency_loss
    )

    # === COMPREHENSIVE METRICS ===
    with torch.no_grad():
        # Basic accuracy metrics
        masked_accuracy = 0.0
        unmasked_accuracy = 0.0

        if masked_logits.numel() > 0:
            masked_preds = masked_logits.argmax(dim=-1)
            if pad_token_id is not None:
                non_padding_mask = masked_targets != pad_token_id
                if non_padding_mask.sum() > 0:
                    masked_accuracy = (
                        (masked_preds[non_padding_mask] == masked_targets[non_padding_mask])
                        .float().mean().item()
                    )
            else:
                masked_accuracy = (masked_preds == masked_targets).float().mean().item()

        if unmasked_logits.numel() > 0:
            unmasked_targets = tokens[~token_mask]
            unmasked_preds = unmasked_logits.argmax(dim=-1)
            if pad_token_id is not None:
                non_padding_mask = unmasked_targets != pad_token_id
                if non_padding_mask.sum() > 0:
                    unmasked_accuracy = (
                        (unmasked_preds[non_padding_mask] == unmasked_targets[non_padding_mask])
                        .float().mean().item()
                    )
            else:
                unmasked_accuracy = (unmasked_preds == unmasked_targets).float().mean().item()

        # Perplexity
        perplexity = torch.exp(token_loss).item() if torch.isfinite(token_loss) else 999.0
        perplexity = min(perplexity, 1e4)

        # DREAM-specific metrics
        dream_metrics = {}
        
        # Bidirectional effectiveness (compare masked vs unmasked accuracy)
        if masked_accuracy > 0 and unmasked_accuracy > 0:
            dream_metrics["bidirectional_effectiveness"] = masked_accuracy / (unmasked_accuracy + 1e-8)
        
        # Skip adaptive noise scheduling metrics for 3D model
        # (3D model doesn't have adaptive_noise_scheduler)
        
        # Skip confidence threshold metrics for 3D model
        # (3D model doesn't have masked_decoder)

        # Enhanced thought metrics
        thought_flat = new_thought.view(batch_size, -1)
        
        # Thought activation analysis
        thought_rms = torch.sqrt(torch.mean(thought_flat**2, dim=1)).mean().item()
        threshold = 0.01 * thought_rms
        active_fraction = (torch.abs(thought_flat) > threshold).float().mean().item()
        
        thought_min = thought_flat.min().item()
        thought_max = thought_flat.max().item()
        dynamic_range = thought_max - thought_min
        
        # Thought consistency within batch
        if batch_size > 1:
            pairwise_sim = F.cosine_similarity(
                thought_flat.unsqueeze(1), thought_flat.unsqueeze(0), dim=2
            )
            mask_eye = ~torch.eye(batch_size, dtype=torch.bool, device=device)
            thought_consistency = pairwise_sim[mask_eye].mean().item()
        else:
            thought_consistency = 1.0

        # Thought evolution metrics - compare with most recent thought in stack
        if thought_stacks.numel() > 0:
            # Get the most recent thought from stack (position 0 in depth dimension)
            latest_in_stack = thought_stacks[:, :, :, :, 0]  # (B, C, H, W)
            evolution_diff = F.mse_loss(new_thought, latest_in_stack, reduction='mean')
            thought_evolution_smoothness = evolution_diff.item()
        else:
            thought_evolution_smoothness = 0.0

        # Skip independence gate analysis for 3D model
        # (3D model doesn't have thought_stack_encoder with independence_gate)
        independence_gate_value = 0.5

        # Masking statistics
        if pad_token_id is not None:
            non_pad_mask = tokens != pad_token_id
            if non_pad_mask.sum() > 0:
                masked_non_pad = (token_mask & non_pad_mask).sum().float()
                total_non_pad = non_pad_mask.sum().float()
                mask_ratio_actual = (masked_non_pad / total_non_pad).item()
            else:
                mask_ratio_actual = 0.0
        else:
            mask_ratio_actual = token_mask.float().mean().item()

    # Compile comprehensive metrics
    metrics = {
        # Core losses
        "total_loss": total_loss.item() if torch.isfinite(total_loss) else 999.0,
        "token_loss": token_loss.item() if torch.isfinite(token_loss) else 999.0,
        "confidence_loss": confidence_loss.item() if torch.isfinite(confidence_loss) else 0.0,
        "thought_loss_total": thought_loss_total.item() if torch.isfinite(thought_loss_total) else 0.0,
        "thought_consistency_loss": thought_consistency_loss.item() if torch.isfinite(thought_consistency_loss) else 0.0,
        
        # Accuracy metrics
        "masked_accuracy": masked_accuracy,
        "unmasked_accuracy": unmasked_accuracy,
        "perplexity": perplexity,
        
        # DREAM-specific metrics
        **dream_metrics,
        
        # Enhanced thought metrics
        "thought_rms": thought_rms,
        "thought_sparsity": 1.0 - active_fraction,
        "thought_dynamic_range": dynamic_range,
        "thought_consistency": thought_consistency,
        "thought_evolution_smoothness": thought_evolution_smoothness,
        "thought_independence_gate": independence_gate_value,
        "thought_mean": new_thought.mean().item(),
        "thought_std": new_thought.std().item(),
        "thought_min": thought_min,
        "thought_max": thought_max,
        "thought_active_fraction": active_fraction,
        
        # Masking statistics
        "mask_ratio_actual": mask_ratio_actual,
        "masked_tokens": token_mask.sum().item(),
        "unmasked_tokens": (~token_mask).sum().item(),
        
        # Training metrics
        "curriculum_weight": curriculum_weight,
    }
    
    # Add individual thought losses to metrics
    for loss_name, loss_value in thought_losses.items():
        metrics[f"thought_{loss_name}"] = loss_value.item() if torch.isfinite(loss_value) else 0.0

    return total_loss, metrics, evolved_thought_stacks