"""
Advanced Thought Analysis Utilities
Provides comprehensive metrics for evaluating thought tensor quality and evolution
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from scipy.stats import pearsonr


class ThoughtAnalyzer:
    """Comprehensive analyzer for thought tensor metrics"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.epsilon = 1e-8  # Small constant to avoid numerical issues
        
        # Metric history for temporal analysis
        self.thought_history = []
        self.prediction_history = []
        self.max_history_length = self.config.get('max_history_length', 100)
    
    def update_history(self, thoughts: torch.Tensor, predictions: Optional[torch.Tensor] = None):
        """Update history buffers for temporal analysis"""
        # Store detached copies to avoid memory leaks
        self.thought_history.append(thoughts.detach().cpu())
        if predictions is not None:
            self.prediction_history.append(predictions.detach().cpu())
        
        # Maintain fixed history size
        if len(self.thought_history) > self.max_history_length:
            self.thought_history.pop(0)
        if len(self.prediction_history) > self.max_history_length:
            self.prediction_history.pop(0)
    
    def compute_information_theoretic_metrics(self, thoughts: torch.Tensor) -> Dict[str, float]:
        """Compute information-theoretic metrics for thoughts"""
        metrics = {}
        
        # Flatten thoughts for analysis
        batch_size = thoughts.shape[0]
        thought_flat = thoughts.view(batch_size, -1)
        
        # 1. Thought Entropy (Shannon entropy of activations)
        # Discretize activations into bins for entropy calculation
        n_bins = 50
        thought_min, thought_max = thought_flat.min(), thought_flat.max()
        if thought_max > thought_min:
            # Normalize to [0, 1] and discretize
            normalized = (thought_flat - thought_min) / (thought_max - thought_min + self.epsilon)
            discretized = (normalized * (n_bins - 1)).long()
            
            # Compute histogram and entropy
            hist = torch.histc(discretized.float(), bins=n_bins, min=0, max=n_bins-1)
            probs = hist / (hist.sum() + self.epsilon)
            probs = probs[probs > 0]  # Remove zero probabilities
            entropy = -(probs * torch.log(probs)).sum().item()
            metrics['thought_entropy'] = entropy
        else:
            metrics['thought_entropy'] = 0.0
        
        # 2. Effective Dimensionality (Participation Ratio)
        # Measures how many dimensions are actively used
        variances = torch.var(thought_flat, dim=0)
        normalized_vars = variances / (variances.sum() + self.epsilon)
        effective_dim = 1.0 / (normalized_vars ** 2).sum().item()
        metrics['effective_dimensionality'] = effective_dim
        
        # 3. Sparsity (fraction of near-zero activations)
        sparsity_threshold = 0.01
        sparse_fraction = (torch.abs(thought_flat) < sparsity_threshold).float().mean().item()
        metrics['thought_sparsity_fraction'] = sparse_fraction
        
        # 4. Activation saturation (fraction of saturated neurons)
        saturation_threshold = 0.95
        max_abs_activation = torch.abs(thought_flat).max(dim=0)[0]
        saturated_fraction = (max_abs_activation > saturation_threshold).float().mean().item()
        metrics['thought_saturation_fraction'] = saturated_fraction
        
        return metrics
    
    def compute_temporal_evolution_metrics(self, current_thoughts: torch.Tensor, 
                                         previous_thoughts: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Compute temporal evolution metrics"""
        metrics = {}
        
        if previous_thoughts is not None:
            # 1. Thought Velocity (rate of change)
            # Flatten the thoughts first, then compute norm
            current_flat = current_thoughts.view(current_thoughts.shape[0], -1)
            previous_flat = previous_thoughts.view(previous_thoughts.shape[0], -1)
            velocity = torch.norm(current_flat - previous_flat, dim=1)
            metrics['thought_velocity_mean'] = velocity.mean().item()
            metrics['thought_velocity_std'] = velocity.std().item()
            
            # 2. Cosine similarity between consecutive thoughts
            cosine_sim = F.cosine_similarity(current_flat, previous_flat, dim=1)
            metrics['thought_temporal_similarity'] = cosine_sim.mean().item()
            
            # 3. KL Divergence between thought distributions
            # Normalize to probability distributions
            current_probs = F.softmax(current_flat, dim=1)
            previous_probs = F.softmax(previous_flat, dim=1)
            
            kl_div = F.kl_div(
                torch.log(current_probs + self.epsilon),
                previous_probs,
                reduction='batchmean'
            )
            metrics['thought_kl_divergence'] = kl_div.item()
        
        # 4. Stack utilization metrics (if thoughts have multiple dimensions)
        if len(current_thoughts.shape) >= 3:  # Has multiple dimensions beyond batch
            # If we can infer a stack structure, analyze it
            if len(current_thoughts.shape) == 4:  # Assume (batch, stack, height, width)
                stack_size = current_thoughts.shape[1]
                
                # Compute norm for each stack position
                stack_norms = torch.norm(
                    current_thoughts.view(current_thoughts.shape[0], stack_size, -1), 
                    dim=2
                )  # (batch, stack)
                
                # Which stack positions are most active?
                mean_stack_activity = stack_norms.mean(dim=0)  # (stack,)
                metrics['stack_max_activity_pos'] = mean_stack_activity.argmax().item()
                metrics['stack_activity_concentration'] = (
                    mean_stack_activity.max() / (mean_stack_activity.mean() + self.epsilon)
                ).item()
                
                # Stack utilization efficiency
                active_positions = (mean_stack_activity > mean_stack_activity.mean() * 0.1).sum().item()
                metrics['stack_utilization_efficiency'] = active_positions / stack_size
            else:
                # Generic dimensionality analysis
                metrics['stack_utilization_efficiency'] = 1.0  # Default for non-stack structures
        
        return metrics
    
    def compute_task_performance_metrics(self, thoughts: torch.Tensor, 
                                       predictions: torch.Tensor,
                                       targets: torch.Tensor) -> Dict[str, float]:
        """Compute task-specific performance metrics"""
        metrics = {}
        
        # 1. Prediction Confidence (entropy of output distribution)
        pred_probs = F.softmax(predictions, dim=-1)
        pred_entropy = -(pred_probs * torch.log(pred_probs + self.epsilon)).sum(dim=-1)
        metrics['prediction_confidence'] = (-pred_entropy.mean()).item()  # Higher is more confident
        
        # 2. Token-level accuracy
        pred_tokens = predictions.argmax(dim=-1)
        accuracy = (pred_tokens == targets).float().mean().item()
        metrics['token_accuracy'] = accuracy
        
        # 3. Thought-Prediction Correlation
        # Compute correlation between thought magnitude and prediction confidence
        thought_magnitude = torch.norm(thoughts.view(thoughts.shape[0], -1), dim=1)
        pred_confidence = -pred_entropy.mean(dim=1)  # Average confidence per sample
        
        if len(thought_magnitude) > 1:  # Need at least 2 samples for correlation
            try:
                corr_coef, _ = pearsonr(
                    thought_magnitude.cpu().numpy(),
                    pred_confidence.cpu().numpy()
                )
                metrics['thought_prediction_correlation'] = corr_coef if not np.isnan(corr_coef) else 0.0
            except:
                metrics['thought_prediction_correlation'] = 0.0
        else:
            metrics['thought_prediction_correlation'] = 0.0
        
        # 4. Contextual Coherence (similarity of thoughts for similar inputs)
        # This is more complex and might require sequence-level analysis
        batch_size = thoughts.shape[0]
        if batch_size > 1:
            thought_flat = thoughts.view(batch_size, -1)
            pairwise_sim = F.cosine_similarity(
                thought_flat.unsqueeze(1), 
                thought_flat.unsqueeze(0), 
                dim=2
            )
            # Average similarity excluding self-similarity
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=thoughts.device)
            avg_similarity = pairwise_sim[mask].mean().item()
            metrics['contextual_coherence'] = avg_similarity
        else:
            metrics['contextual_coherence'] = 1.0
        
        return metrics
    
    def compute_stability_metrics(self, thoughts: torch.Tensor) -> Dict[str, float]:
        """Compute stability and robustness metrics"""
        metrics = {}
        
        batch_size = thoughts.shape[0]
        if batch_size < 2:
            return metrics
        
        thought_flat = thoughts.view(batch_size, -1)
        
        # 1. Inter-batch stability (how consistent are thoughts across batch)
        stability = -torch.var(thought_flat, dim=0).mean().item()  # Lower variance = higher stability
        metrics['thought_stability'] = stability
        
        # 2. Thought clustering coefficient
        # Measure how much thoughts cluster together
        pairwise_distances = torch.cdist(thought_flat, thought_flat)
        mean_distance = pairwise_distances.mean().item()
        metrics['thought_clustering'] = -mean_distance  # Lower distance = higher clustering
        
        # 3. Dynamic range utilization
        thought_min = thought_flat.min().item()
        thought_max = thought_flat.max().item()
        dynamic_range = thought_max - thought_min
        metrics['dynamic_range'] = dynamic_range
        
        return metrics
    
    def compute_convergence_metrics(self) -> Dict[str, float]:
        """Compute convergence metrics using historical data"""
        metrics = {}
        
        if len(self.thought_history) < 3:
            return metrics
        
        # Convert recent history to tensors
        recent_thoughts = torch.stack(self.thought_history[-10:])  # Last 10 steps
        seq_length = recent_thoughts.shape[0]
        
        if seq_length < 3:
            return metrics
        
        # 1. Convergence rate (how quickly thoughts stabilize)
        thought_flat = recent_thoughts.view(seq_length, -1)
        differences = torch.norm(thought_flat[1:] - thought_flat[:-1], dim=1)
        
        # Fit exponential decay to differences
        if len(differences) > 2:
            # Simple convergence measure: ratio of recent to initial change
            recent_change = differences[-1].item()
            initial_change = differences[0].item()
            convergence_ratio = recent_change / (initial_change + self.epsilon)
            metrics['convergence_rate'] = -convergence_ratio  # Lower ratio = faster convergence
        
        # 2. Oscillation detection
        # Look for periodic patterns in thought changes
        if len(differences) >= 4:
            # Simple oscillation detection: correlation between differences
            diffs_np = differences.cpu().numpy()
            even_diffs = diffs_np[::2]
            odd_diffs = diffs_np[1::2]
            
            min_len = min(len(even_diffs), len(odd_diffs))
            if min_len > 1:
                try:
                    oscillation_corr, _ = pearsonr(even_diffs[:min_len], odd_diffs[:min_len])
                    metrics['thought_oscillation'] = oscillation_corr if not np.isnan(oscillation_corr) else 0.0
                except:
                    metrics['thought_oscillation'] = 0.0
        
        return metrics
    
    def compute_all_metrics(self, current_thoughts: torch.Tensor,
                          previous_thoughts: Optional[torch.Tensor] = None,
                          predictions: Optional[torch.Tensor] = None,
                          targets: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Compute all available metrics"""
        all_metrics = {}
        
        # Information-theoretic metrics
        info_metrics = self.compute_information_theoretic_metrics(current_thoughts)
        all_metrics.update(info_metrics)
        
        # Temporal evolution metrics
        temporal_metrics = self.compute_temporal_evolution_metrics(current_thoughts, previous_thoughts)
        all_metrics.update(temporal_metrics)
        
        # Task performance metrics
        if predictions is not None and targets is not None:
            task_metrics = self.compute_task_performance_metrics(current_thoughts, predictions, targets)
            all_metrics.update(task_metrics)
        
        # Stability metrics  
        stability_metrics = self.compute_stability_metrics(current_thoughts)
        all_metrics.update(stability_metrics)
        
        # Convergence metrics (using history)
        convergence_metrics = self.compute_convergence_metrics()
        all_metrics.update(convergence_metrics)
        
        # Update history for next iteration
        self.update_history(current_thoughts, predictions)
        
        return all_metrics


def create_thought_analyzer(config: Optional[Dict] = None) -> ThoughtAnalyzer:
    """Factory function to create a ThoughtAnalyzer"""
    return ThoughtAnalyzer(config)