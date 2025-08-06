"""
Enhanced Thought System with Self-Learning Temporal Dynamics
Implements 3D→2D thought evolution with self-supervised learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict


class TemporalAttention(nn.Module):
    """Multi-scale temporal attention for thought stack processing"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, num_scales: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_scales = num_scales
        
        # Multi-scale temporal attention
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads // num_scales, 
                                 dropout=0.1, batch_first=True)
            for _ in range(num_scales)
        ])
        
        # Learnable scale mixing
        self.scale_mixer = nn.Linear(hidden_dim * num_scales, hidden_dim)
        
        # Importance scoring for each temporal position
        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)  # Normalize across time dimension
        )
    
    def forward(self, thought_stack: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process thought stack with multi-scale temporal attention
        Args:
            thought_stack: (batch, time, hidden_dim)
        Returns:
            attended: (batch, hidden_dim)
            importance_weights: (batch, time)
        """
        _, time_steps, _ = thought_stack.shape
        
        # Multi-scale processing
        scale_outputs = []
        
        for scale_idx, attention in enumerate(self.scale_attentions):
            # Different scales focus on different temporal ranges
            if scale_idx == 0:  # Fine scale - recent thoughts
                scale_input = thought_stack[:, -min(3, time_steps):, :]
            elif scale_idx == 1:  # Medium scale - middle thoughts
                scale_input = thought_stack[:, -min(6, time_steps)::2, :]  # Every 2nd
            else:  # Coarse scale - all thoughts
                scale_input = thought_stack[:, ::3, :]  # Every 3rd
            
            # Apply attention at this scale
            query = thought_stack[:, -1:, :]  # Use latest thought as query
            attended, _ = attention(query, scale_input, scale_input)
            scale_outputs.append(attended.squeeze(1))
        
        # Combine scales
        multi_scale = torch.cat(scale_outputs, dim=-1)
        combined = self.scale_mixer(multi_scale)
        
        # Compute importance weights for the full stack
        importance_weights = self.importance_scorer(thought_stack).squeeze(-1)
        
        # Apply importance weighting
        weighted_stack = thought_stack * importance_weights.unsqueeze(-1)
        final_output = weighted_stack.sum(dim=1)
        
        # Combine with multi-scale attention
        output = 0.5 * combined + 0.5 * final_output
        
        return output, importance_weights


class TemporalDynamics(nn.Module):
    """Compute temporal derivatives and dynamics of thought evolution"""
    
    def __init__(self, thought_dims: Tuple[int, ...]):
        super().__init__()
        self.thought_dims = thought_dims
        thought_size = math.prod(thought_dims)
        
        # Velocity and acceleration encoders
        self.velocity_encoder = nn.Linear(thought_size, thought_size)
        self.acceleration_encoder = nn.Linear(thought_size, thought_size)
        
        # Dynamics mixer
        self.dynamics_mixer = nn.Sequential(
            nn.Linear(thought_size * 3, thought_size * 2),
            nn.GELU(),
            nn.Linear(thought_size * 2, thought_size)
        )
    
    def forward(self, thought_stack: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute temporal dynamics
        Args:
            thought_stack: (batch, time, *thought_dims)
        Returns:
            Dictionary with position, velocity, acceleration
        """
        batch_size, time_steps = thought_stack.shape[:2]
        
        # Flatten thoughts for processing
        stack_flat = thought_stack.view(batch_size, time_steps, -1)
        
        # Compute velocity (first derivative)
        if time_steps > 1:
            velocity = stack_flat[:, 1:] - stack_flat[:, :-1]
            # Pad to maintain shape
            velocity = F.pad(velocity, (0, 0, 1, 0))
        else:
            velocity = torch.zeros_like(stack_flat)
        
        # Compute acceleration (second derivative)
        if time_steps > 2:
            accel = velocity[:, 1:] - velocity[:, :-1]
            accel = F.pad(accel, (0, 0, 1, 0))
        else:
            accel = torch.zeros_like(stack_flat)
        
        # Encode dynamics
        velocity_encoded = self.velocity_encoder(velocity)
        accel_encoded = self.acceleration_encoder(accel)
        
        # Combine position, velocity, acceleration
        dynamics = torch.cat([stack_flat, velocity_encoded, accel_encoded], dim=-1)
        mixed = self.dynamics_mixer(dynamics)
        
        return {
            'position': stack_flat,
            'velocity': velocity_encoded,
            'acceleration': accel_encoded,
            'combined': mixed
        }


class SelfSupervisedThoughtLearning(nn.Module):
    """Self-supervised learning objectives for thought evolution"""
    
    def __init__(self, thought_dims: Tuple[int, ...], hidden_dim: int = 512):
        super().__init__()
        self.thought_dims = thought_dims
        thought_size = math.prod(thought_dims)
        
        # Thought predictor for next-thought prediction
        self.thought_predictor = nn.Sequential(
            nn.Linear(thought_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, thought_size)
        )
        
        # Thought reconstructor for autoencoding
        self.encoder = nn.Sequential(
            nn.Linear(thought_size, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, thought_size)
        )
        
        # Contrastive projection head
        self.projection_head = nn.Sequential(
            nn.Linear(thought_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128)  # Project to lower dim for contrastive learning
        )
    
    def forward(self, thought_stack: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute self-supervised objectives
        Args:
            thought_stack: (batch, time, *thought_dims)
        Returns:
            Dictionary with predictions and reconstructions
        """
        batch_size, time_steps = thought_stack.shape[:2]
        stack_flat = thought_stack.view(batch_size, time_steps, -1)
        
        results = {}
        
        # Next-thought prediction
        if time_steps > 1:
            # Predict each thought from the previous one
            predictions = []
            for t in range(time_steps - 1):
                pred = self.thought_predictor(stack_flat[:, t])
                predictions.append(pred)
            results['next_predictions'] = torch.stack(predictions, dim=1)
            results['next_targets'] = stack_flat[:, 1:]
        
        # Autoencoding
        encoded = self.encoder(stack_flat)
        reconstructed = self.decoder(encoded)
        results['reconstructed'] = reconstructed
        results['encoded'] = encoded
        
        # Contrastive embeddings
        projections = self.projection_head(stack_flat)
        results['projections'] = projections
        
        return results
    
    def compute_losses(self, results: Dict[str, torch.Tensor], 
                      original_stack: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute self-supervised losses"""
        losses = {}
        
        # Next-thought prediction loss
        if 'next_predictions' in results:
            pred_loss = F.mse_loss(results['next_predictions'], results['next_targets'])
            losses['prediction_loss'] = pred_loss
        
        # Reconstruction loss
        batch_size, time_steps = original_stack.shape[:2]
        stack_flat = original_stack.view(batch_size, time_steps, -1)
        recon_loss = F.mse_loss(results['reconstructed'], stack_flat)
        losses['reconstruction_loss'] = recon_loss
        
        # Contrastive loss (SimCLR-style)
        projections = results['projections']
        if projections.shape[1] > 1:
            # Positive pairs: consecutive thoughts
            positives = F.cosine_similarity(
                projections[:, :-1].reshape(-1, projections.shape[-1]),
                projections[:, 1:].reshape(-1, projections.shape[-1])
            )
            
            # Negative pairs: random thoughts from different positions
            batch_size, time_steps, proj_dim = projections.shape
            proj_flat = projections.reshape(-1, proj_dim)
            
            # Random negatives
            perm = torch.randperm(proj_flat.shape[0])
            negatives = F.cosine_similarity(proj_flat, proj_flat[perm])
            
            # Contrastive loss: push positives together, negatives apart
            contrastive_loss = -positives.mean() + negatives.mean()
            losses['contrastive_loss'] = contrastive_loss
        
        return losses


class EnhancedThoughtStackEncoder(nn.Module):
    """Enhanced 3D→2D thought encoder with self-learning capabilities"""
    
    def __init__(self, thought_dims: Tuple[int, ...], stack_size: int, 
                 hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.thought_dims = thought_dims
        self.stack_size = stack_size
        self.hidden_dim = hidden_dim
        
        single_thought_size = math.prod(thought_dims)
        
        # Original thought processor
        self.thought_processor = nn.Sequential(
            nn.Linear(single_thought_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Enhanced temporal components
        self.temporal_attention = TemporalAttention(hidden_dim)
        self.temporal_dynamics = TemporalDynamics(thought_dims)
        self.self_supervised = SelfSupervisedThoughtLearning(thought_dims, hidden_dim)
        
        # Bidirectional temporal processing
        self.forward_lstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True)
        self.backward_lstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True)
        
        # Hierarchical temporal abstraction
        self.temporal_hierarchy = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(3)
        ])
        
        # Learnable temporal positional encoding
        self.temporal_pos_encoding = nn.Parameter(
            torch.randn(1, stack_size, hidden_dim) * 0.02
        )
        
        # Sinusoidal frequency components for rhythm learning
        self.register_buffer('freq_bands', 
                           torch.linspace(0.1, 10, hidden_dim // 2))
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # 3x for multiple streams
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Dropout(dropout / 2)
        )
        
        # Gradient isolation gate (for independent learning)
        self.independence_gate = nn.Parameter(torch.tensor(0.5))
    
    def add_temporal_encoding(self, thought_stack: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal temporal encoding for rhythm learning"""
        _, time_steps, _ = thought_stack.shape
        
        # Create position indices
        positions = torch.arange(time_steps, device=thought_stack.device).float()
        positions = positions.unsqueeze(0).unsqueeze(-1)  # (1, time, 1)
        
        # Apply sinusoidal encoding
        angles = positions * self.freq_bands.unsqueeze(0).unsqueeze(0)
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)
        
        temporal_enc = torch.cat([sin_enc, cos_enc], dim=-1)[:, :, :hidden_dim]
        
        return thought_stack + 0.1 * temporal_enc
    
    def forward(self, thought_stack: torch.Tensor, 
                return_self_supervised: bool = False) -> torch.Tensor:
        """
        Enhanced 3D→2D encoding with self-learning
        Args:
            thought_stack: (batch_size, stack_size, *thought_dims)
            return_self_supervised: Whether to return self-supervised results
        Returns:
            encoded: (batch_size, output_dim)
        """
        batch_size, stack_size = thought_stack.shape[:2]
        
        # Process thoughts
        thought_flat = thought_stack.view(batch_size * stack_size, -1)
        processed = self.thought_processor(thought_flat)
        processed = processed.view(batch_size, stack_size, -1)
        
        # Add positional and temporal encoding
        processed = processed + self.temporal_pos_encoding[:, :stack_size, :]
        processed = self.add_temporal_encoding(processed)
        
        # Temporal attention processing
        attended, _ = self.temporal_attention(processed)
        
        # Bidirectional temporal processing
        forward_out, _ = self.forward_lstm(processed)
        backward_out, _ = self.backward_lstm(torch.flip(processed, [1]))
        backward_out = torch.flip(backward_out, [1])
        bidirectional = torch.cat([forward_out[:, -1], backward_out[:, -1]], dim=-1)
        
        # Temporal dynamics
        _ = self.temporal_dynamics(thought_stack)
        # dynamics_encoded = dynamics['combined'][:, -1]  # Take last timestep (unused)
        
        # Hierarchical processing (different time scales)
        hierarchical_outputs = []
        for i, layer in enumerate(self.temporal_hierarchy):
            if i == 0:  # Recent
                h_input = processed[:, -min(2, stack_size):].mean(dim=1)
            elif i == 1:  # Medium
                h_input = processed[:, -min(4, stack_size):].mean(dim=1)
            else:  # All
                h_input = processed.mean(dim=1)
            hierarchical_outputs.append(layer(h_input))
        hierarchical = torch.stack(hierarchical_outputs).mean(dim=0)
        
        # Combine all representations
        combined = torch.cat([attended, bidirectional, hierarchical], dim=-1)
        
        # Apply independence gate (for gradient isolation)
        if self.training:
            # During training, partially isolate gradients
            combined = self.independence_gate * combined.detach() + \
                      (1 - self.independence_gate) * combined
        
        # Final projection
        output = self.output_projection(combined)
        
        if return_self_supervised:
            # Compute self-supervised objectives
            self_supervised_results = self.self_supervised(thought_stack)
            return output, self_supervised_results
        
        return output


class EnhancedThoughtStack:
    """Enhanced thought stack with importance-based retention"""
    
    def __init__(self, stack_size: int, thought_dims: Tuple[int, ...], 
                 device: str = 'cuda'):
        self.stack_size = stack_size
        self.thought_dims = thought_dims
        self.device = device
        
        # Track importance scores for each position
        self.importance_scores = None
    
    def initialize_stack(self, batch_size: int, noise_std: float = 0.1) -> torch.Tensor:
        """Initialize stack with small noise"""
        stack = torch.randn(
            batch_size, self.stack_size, *self.thought_dims, 
            device=self.device
        ) * noise_std
        
        # Reset importance scores
        self.importance_scores = torch.ones(
            batch_size, self.stack_size, device=self.device
        ) / self.stack_size
        
        return stack
    
    def push_thought_with_importance(self, stack: torch.Tensor, 
                                    new_thought: torch.Tensor,
                                    importance_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add new thought with importance-based retention
        Args:
            stack: (batch_size, stack_size, *thought_dims)
            new_thought: (batch_size, *thought_dims)
            importance_weights: (batch_size, stack_size) - importance of each position
        Returns:
            updated_stack: (batch_size, stack_size, *thought_dims)
        """
        if importance_weights is None:
            # Standard FIFO if no importance weights
            return torch.cat([stack[:, 1:], new_thought.unsqueeze(1)], dim=1)
        
        # Find least important position to replace
        batch_size = stack.shape[0]
        updated_stack = stack.clone()
        
        for b in range(batch_size):
            # Find position with lowest importance
            min_importance_idx = torch.argmin(importance_weights[b])
            
            # Replace least important thought
            updated_stack[b, min_importance_idx] = new_thought[b]
        
        # Update importance scores
        self.importance_scores = importance_weights
        
        return updated_stack


if __name__ == "__main__":
    # Test the enhanced thought system
    print("🧠 Testing Enhanced Thought System")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    stack_size = 8
    thought_dims = (16, 16, 8)
    hidden_dim = 256
    output_dim = 512
    
    # Create enhanced encoder
    encoder = EnhancedThoughtStackEncoder(
        thought_dims, stack_size, hidden_dim, output_dim
    ).to(device)
    
    # Create test thought stack
    thought_stack = torch.randn(batch_size, stack_size, *thought_dims).to(device)
    
    # Test forward pass
    encoded, self_supervised = encoder(thought_stack, return_self_supervised=True)
    
    print(f"✅ Encoded shape: {encoded.shape}")
    print(f"✅ Self-supervised results: {self_supervised.keys()}")
    
    # Compute self-supervised losses
    losses = encoder.self_supervised.compute_losses(self_supervised, thought_stack)
    print(f"✅ Self-supervised losses: {losses.keys()}")
    
    for loss_name, loss_value in losses.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")
    
    print("\n🎉 Enhanced thought system test complete!")