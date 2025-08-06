#!/usr/bin/env python3
"""
DREAM-Specific Metrics Analyzer (Lite Version)
Analyzes thought evolution, cross-attention patterns, and DREAM-specific improvements
No scipy dependency for compatibility
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


class ThoughtAnalyzer:
    """Analyzes thought evolution and DREAM-specific patterns"""
    
    def __init__(self):
        self.thought_history = []
        self.attention_patterns = []
        self.previous_stack = None
        
    def analyze_thought_evolution(self, 
                                current_stack: torch.Tensor, 
                                new_thought: Optional[torch.Tensor] = None) -> Dict:
        """
        Analyze how thoughts evolve between forward passes
        Args:
            current_stack: (B, C, H, W, D) current thought stack
            new_thought: (B, C, H, W) newly generated thought
        Returns:
            Dict with evolution metrics
        """
        metrics = {}
        
        if self.previous_stack is not None:
            # Calculate evolution rate - flatten last 4 dimensions for norm calculation
            curr_flat = current_stack.reshape(current_stack.shape[0], -1)
            prev_flat = self.previous_stack.reshape(self.previous_stack.shape[0], -1)
            stack_diff = torch.norm(curr_flat - prev_flat, dim=1)
            metrics['evolution_rate'] = stack_diff.mean().item()
            metrics['evolution_std'] = stack_diff.std().item()
            
            # Calculate per-depth evolution
            depth_evolution = []
            for d in range(current_stack.shape[-1]):
                curr_depth = current_stack[:,:,:,:,d].reshape(current_stack.shape[0], -1)
                prev_depth = self.previous_stack[:,:,:,:,d].reshape(self.previous_stack.shape[0], -1)
                depth_diff = torch.norm(curr_depth - prev_depth, dim=1).mean().item()
                depth_evolution.append(depth_diff)
            
            metrics['depth_evolution'] = depth_evolution
            metrics['most_evolved_depth'] = int(np.argmax(depth_evolution))
            metrics['least_evolved_depth'] = int(np.argmin(depth_evolution))
            
            # Calculate evolution consistency (lower std = more consistent)
            metrics['evolution_consistency'] = 1.0 - (np.std(depth_evolution) / (np.mean(depth_evolution) + 1e-8))
        
        # Analyze current stack properties
        metrics['stack_sparsity'] = (torch.abs(current_stack) < 0.01).float().mean().item()
        metrics['stack_rms'] = torch.sqrt(torch.mean(current_stack ** 2)).item()
        metrics['stack_dynamic_range'] = (current_stack.max() - current_stack.min()).item()
        
        # Update history
        if len(self.thought_history) >= 10:  # Keep last 10 stacks
            self.thought_history.pop(0)
        self.thought_history.append(current_stack.detach().cpu())
        
        self.previous_stack = current_stack.detach()
        
        return metrics
    
    def analyze_cross_attention_entropy(self, attention_weights: torch.Tensor) -> Dict:
        """
        Analyze diversity of cross-attention between thoughts and tokens
        Args:
            attention_weights: Attention weights from cross-attention layers
        Returns:
            Dict with attention entropy metrics
        """
        if attention_weights is None:
            return {'attention_entropy': 0.0, 'attention_diversity': 0.0}
        
        # Calculate entropy of attention distribution
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        attention_probs = torch.softmax(attention_weights + epsilon, dim=-1)
        entropy = -torch.sum(attention_probs * torch.log(attention_probs + epsilon), dim=-1)
        
        metrics = {
            'attention_entropy': entropy.mean().item(),
            'attention_entropy_std': entropy.std().item(),
            'attention_diversity': entropy.max().item() - entropy.min().item(),
            'attention_uniformity': 1.0 - (entropy.std() / (entropy.mean() + epsilon)).item()
        }
        
        return metrics
    
    def analyze_thought_token_correlation(self, 
                                        thought_features: torch.Tensor,
                                        token_predictions: torch.Tensor,
                                        targets: torch.Tensor) -> Dict:
        """
        Analyze correlation between thought patterns and token prediction success
        Args:
            thought_features: (B, embed_dim) aggregated thought features
            token_predictions: (B, seq_len) predicted token IDs
            targets: (B, seq_len) target token IDs
        Returns:
            Dict with correlation metrics
        """
        B, seq_len = token_predictions.shape
        
        # Calculate prediction accuracy per sample
        correct_predictions = (token_predictions == targets).float().mean(dim=1)  # (B,)
        
        # Calculate correlation between thought features and accuracy
        correlations = []
        
        # For each dimension of thought features
        for dim in range(thought_features.shape[1]):
            thought_dim = thought_features[:, dim].cpu().numpy()
            accuracy = correct_predictions.cpu().numpy()
            
            if len(set(accuracy)) > 1:  # Need variance in accuracy for correlation
                # Simple correlation using numpy
                corr_matrix = np.corrcoef(thought_dim, accuracy)
                if corr_matrix.shape == (2, 2):
                    corr = corr_matrix[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        if correlations:
            metrics = {
                'max_correlation': max(correlations),
                'avg_correlation': np.mean(correlations),
                'correlation_consistency': 1.0 - (np.std(correlations) / (np.mean(correlations) + 1e-8))
            }
        else:
            metrics = {
                'max_correlation': 0.0,
                'avg_correlation': 0.0,
                'correlation_consistency': 0.0
            }
        
        return metrics
    
    def analyze_progressive_unmasking_effectiveness(self,
                                                  logits: torch.Tensor,
                                                  tokens: torch.Tensor,
                                                  initial_mask: torch.Tensor,
                                                  confidence_threshold: float = 0.75) -> Dict:
        """
        Simulate progressive unmasking and measure effectiveness
        Args:
            logits: (B, seq_len, vocab_size) model predictions
            tokens: (B, seq_len) target tokens
            initial_mask: (B, seq_len) initial masking pattern
            confidence_threshold: Threshold for unmasking tokens
        Returns:
            Dict with progressive unmasking metrics
        """
        B, seq_len, vocab_size = logits.shape
        
        # Get prediction confidence (softmax probabilities)
        probs = torch.softmax(logits, dim=-1)
        predicted_tokens = logits.argmax(dim=-1)
        
        # Get confidence for predicted tokens
        confidence_scores = probs.gather(2, predicted_tokens.unsqueeze(-1)).squeeze(-1)
        
        # Simulate progressive unmasking
        current_mask = initial_mask.clone()
        unmasking_steps = []
        accuracy_progression = []
        
        for step in range(5):  # 5 unmasking steps
            # Find high-confidence masked tokens
            high_confidence = (confidence_scores > confidence_threshold) & current_mask
            
            # Unmask high-confidence tokens
            current_mask = current_mask & ~high_confidence
            
            # Calculate accuracy for currently unmasked tokens
            unmasked_predictions = predicted_tokens[~current_mask]
            unmasked_targets = tokens[~current_mask]
            
            if unmasked_predictions.numel() > 0:
                accuracy = (unmasked_predictions == unmasked_targets).float().mean().item()
            else:
                accuracy = 0.0
            
            unmasked_ratio = (~current_mask).float().mean().item()
            
            unmasking_steps.append({
                'step': step,
                'unmasked_ratio': unmasked_ratio,
                'accuracy': accuracy,
                'tokens_unmasked': high_confidence.sum().item()
            })
            accuracy_progression.append(accuracy)
            
            # Lower threshold for next step
            confidence_threshold *= 0.9
        
        # Calculate progressive unmasking effectiveness
        if len(accuracy_progression) > 1:
            accuracy_improvement = accuracy_progression[-1] - accuracy_progression[0]
            unmasking_efficiency = accuracy_improvement / max(unmasking_steps[-1]['unmasked_ratio'], 0.01)
        else:
            accuracy_improvement = 0.0
            unmasking_efficiency = 0.0
        
        metrics = {
            'progressive_accuracy_improvement': accuracy_improvement,
            'unmasking_efficiency': unmasking_efficiency,
            'final_unmasked_ratio': unmasking_steps[-1]['unmasked_ratio'],
            'final_accuracy': accuracy_progression[-1],
            'unmasking_steps': unmasking_steps
        }
        
        return metrics
    
    def compute_comprehensive_metrics(self,
                                    thought_stack: torch.Tensor,
                                    logits: torch.Tensor,
                                    tokens: torch.Tensor,
                                    token_mask: torch.Tensor,
                                    attention_weights: Optional[torch.Tensor] = None) -> Dict:
        """
        Compute all thought-specific metrics in one call
        Args:
            thought_stack: (B, C, H, W, D) current thought stack
            logits: (B, seq_len, vocab_size) model predictions
            tokens: (B, seq_len) target tokens
            token_mask: (B, seq_len) masking pattern
            attention_weights: Optional cross-attention weights
        Returns:
            Comprehensive metrics dictionary
        """
        all_metrics = {}
        
        # Thought evolution analysis
        evolution_metrics = self.analyze_thought_evolution(thought_stack)
        all_metrics.update({f'thought_{k}': v for k, v in evolution_metrics.items()})
        
        # Cross-attention analysis
        if attention_weights is not None:
            attention_metrics = self.analyze_cross_attention_entropy(attention_weights)
            all_metrics.update({f'attention_{k}': v for k, v in attention_metrics.items()})
        
        # Progressive unmasking analysis
        progressive_metrics = self.analyze_progressive_unmasking_effectiveness(
            logits, tokens, token_mask
        )
        all_metrics.update({f'progressive_{k}': v for k, v in progressive_metrics.items()})
        
        # Thought-token correlation (simplified version)
        # Use mean of thought stack as thought features
        thought_features = thought_stack.mean(dim=(2, 3, 4))  # (B, C)
        correlation_metrics = self.analyze_thought_token_correlation(
            thought_features, logits.argmax(dim=-1), tokens
        )
        all_metrics.update({f'correlation_{k}': v for k, v in correlation_metrics.items()})
        
        return all_metrics
    
    def reset(self):
        """Reset analyzer state"""
        self.thought_history.clear()
        self.attention_patterns.clear()
        self.previous_stack = None


def compare_dream_effectiveness(thought_metrics: Dict, baseline_metrics: Dict) -> Dict:
    """
    Compare DREAM effectiveness between thought and baseline models
    Args:
        thought_metrics: Metrics from thought model
        baseline_metrics: Metrics from baseline model
    Returns:
        Comparison analysis
    """
    comparison = {}
    
    # Core DREAM metrics comparison
    if 'masked_accuracy' in thought_metrics and 'masked_accuracy' in baseline_metrics:
        masked_improvement = thought_metrics['masked_accuracy'] - baseline_metrics['masked_accuracy']
        unmasked_improvement = thought_metrics.get('unmasked_accuracy', 0) - baseline_metrics.get('unmasked_accuracy', 0)
        
        comparison.update({
            'masked_accuracy_improvement': masked_improvement,
            'unmasked_accuracy_improvement': unmasked_improvement,
            'dream_specificity': masked_improvement - unmasked_improvement,  # How much more thoughts help masked vs unmasked
            'overall_dream_benefit': (masked_improvement + unmasked_improvement) / 2
        })
        
        # Bidirectional effectiveness comparison
        thought_bidir = thought_metrics.get('bidirectional_effectiveness', 1.0)
        baseline_bidir = baseline_metrics.get('bidirectional_effectiveness', 1.0)
        comparison['bidirectional_improvement'] = thought_bidir - baseline_bidir
        
        # Interpretation
        if masked_improvement > 0.02 and masked_improvement > unmasked_improvement:
            comparison['interpretation'] = "Thoughts significantly improve DREAM's masked prediction capability"
            comparison['recommendation'] = "Continue developing thought system - clear DREAM benefit"
        elif masked_improvement > 0:
            comparison['interpretation'] = "Thoughts provide moderate DREAM improvements"
            comparison['recommendation'] = "Optimize thought system for better masked token performance"
        else:
            comparison['interpretation'] = "No clear DREAM-specific benefit from thoughts"
            comparison['recommendation'] = "Reconsider thought architecture - may not align with DREAM goals"
    
    return comparison