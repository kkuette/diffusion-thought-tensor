"""
Phase 3: Thought Evolution Analysis Tools

This module implements comprehensive analysis tools for understanding how thoughts
evolve during the diffusion denoising process.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class ThoughtSnapshot:
    """Single snapshot of thought state during denoising"""
    timestep: int
    thought_tensor: torch.Tensor
    text_embedding: torch.Tensor
    loss_components: Dict[str, float]
    coherence_score: float
    diversity_score: float
    alignment_score: float

class ThoughtEvolutionAnalyzer:
    """
    Analyzes how thoughts evolve during diffusion denoising process.
    Tracks patterns, coherence, diversity, and emergent behaviors.
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.history = []
        
    def analyze_thought_evolution(self, 
                                initial_thought: torch.Tensor,
                                initial_text: torch.Tensor,
                                num_steps: int = 500,
                                sample_interval: int = 10) -> List[ThoughtSnapshot]:
        """
        Track how thoughts evolve during denoising process.
        
        Args:
            initial_thought: Starting thought tensor (usually noise)
            initial_text: Starting text embeddings (usually noise)
            num_steps: Total denoising steps
            sample_interval: How often to sample (every N steps)
            
        Returns:
            List of ThoughtSnapshot objects tracking evolution
        """
        self.model.eval()
        thought_history = []
        
        current_thought = initial_thought.clone()
        current_text = initial_text.clone()
        
        with torch.no_grad():
            for t in reversed(range(num_steps)):
                timestep = torch.tensor([t], device=self.device)
                
                # Perform denoising step
                if hasattr(self.model, 'reverse_diffusion_step'):
                    current_thought, current_text = self.model.reverse_diffusion_step(
                        current_thought, current_text, timestep
                    )
                else:
                    # Fallback to model forward pass
                    outputs = self.model(current_text, current_thought, timestep)
                    if isinstance(outputs, tuple):
                        current_text, current_thought = outputs
                    else:
                        current_text = outputs
                
                # Sample at specified intervals
                if t % sample_interval == 0 or t == 0:
                    # Compute analysis metrics
                    coherence = self._compute_coherence(current_thought)
                    diversity = self._compute_diversity(current_thought)
                    alignment = self._compute_alignment(current_thought, current_text)
                    
                    # Create snapshot
                    snapshot = ThoughtSnapshot(
                        timestep=t,
                        thought_tensor=current_thought.clone(),
                        text_embedding=current_text.clone(),
                        loss_components={},  # Will be computed separately
                        coherence_score=coherence,
                        diversity_score=diversity,
                        alignment_score=alignment
                    )
                    
                    thought_history.append(snapshot)
        
        self.history = thought_history
        return thought_history
    
    def _compute_coherence(self, thought_tensor: torch.Tensor) -> float:
        """
        Compute thought coherence score based on internal consistency.
        Higher scores indicate more structured/coherent thoughts.
        """
        # Flatten thought tensor for analysis
        flat_thought = thought_tensor.flatten(1)  # Keep batch dim
        
        # Compute pairwise correlations
        if flat_thought.size(1) > 1:
            corr_matrix = torch.corrcoef(flat_thought.T)
            # Remove NaN values and diagonal
            mask = ~torch.isnan(corr_matrix) & ~torch.eye(corr_matrix.size(0), dtype=bool)
            if mask.sum() > 0:
                coherence = corr_matrix[mask].abs().mean().item()
            else:
                coherence = 0.0
        else:
            coherence = 0.0
            
        return coherence
    
    def _compute_diversity(self, thought_tensor: torch.Tensor) -> float:
        """
        Compute thought diversity score based on entropy and variance.
        Higher scores indicate more diverse/creative thoughts.
        """
        flat_thought = thought_tensor.flatten(1)
        
        # Compute variance across dimensions
        variance = torch.var(flat_thought, dim=1).mean().item()
        
        # Normalize to 0-1 range
        diversity = min(1.0, variance / 10.0)  # Scale factor based on typical ranges
        
        return diversity
    
    def _compute_alignment(self, thought_tensor: torch.Tensor, text_embedding: torch.Tensor) -> float:
        """
        Compute alignment between thought and text representations.
        Higher scores indicate better thought-text correspondence.
        """
        # Average pool text embeddings to match thought dimensionality
        if len(text_embedding.shape) == 3:  # [batch, seq, embed]
            text_pooled = text_embedding.mean(dim=1)  # [batch, embed]
        else:
            text_pooled = text_embedding
        
        # Average pool thought tensor
        thought_pooled = thought_tensor.flatten(1).mean(dim=1)  # [batch]
        
        # Expand thought to match text dimensionality if needed
        if thought_pooled.size(-1) != text_pooled.size(-1):
            # Simple projection for alignment computation
            min_dim = min(thought_pooled.size(-1), text_pooled.size(-1))
            thought_pooled = thought_pooled[:, :min_dim] if thought_pooled.dim() > 1 else thought_pooled[:min_dim]
            text_pooled = text_pooled[:, :min_dim]
        
        # Compute cosine similarity
        if thought_pooled.dim() == 1:
            thought_pooled = thought_pooled.unsqueeze(0)
        if text_pooled.dim() == 1:
            text_pooled = text_pooled.unsqueeze(0)
            
        similarity = torch.cosine_similarity(thought_pooled, text_pooled, dim=-1)
        alignment = similarity.mean().item()
        
        return alignment
    
    def detect_thought_patterns(self, min_pattern_length: int = 5) -> Dict[str, List]:
        """
        Detect recurring patterns in thought evolution.
        
        Args:
            min_pattern_length: Minimum length for pattern detection
            
        Returns:
            Dictionary of detected patterns and their characteristics
        """
        if len(self.history) < min_pattern_length:
            return {"patterns": [], "message": "Insufficient history for pattern detection"}
        
        patterns = {
            "coherence_trends": [],
            "diversity_cycles": [],
            "alignment_patterns": [],
            "critical_transitions": []
        }
        
        # Extract time series
        coherence_series = [snap.coherence_score for snap in self.history]
        diversity_series = [snap.diversity_score for snap in self.history]
        alignment_series = [snap.alignment_score for snap in self.history]
        
        # Detect coherence trends
        coherence_trend = self._detect_trend(coherence_series)
        patterns["coherence_trends"].append({
            "type": coherence_trend,
            "strength": self._compute_trend_strength(coherence_series)
        })
        
        # Detect diversity cycles
        diversity_cycles = self._detect_cycles(diversity_series)
        patterns["diversity_cycles"] = diversity_cycles
        
        # Detect critical transitions (sudden changes)
        critical_points = self._detect_critical_transitions(coherence_series, threshold=0.1)
        patterns["critical_transitions"] = critical_points
        
        return patterns
    
    def _detect_trend(self, series: List[float]) -> str:
        """Detect overall trend in a time series"""
        if len(series) < 2:
            return "insufficient_data"
        
        # Simple linear regression
        x = np.arange(len(series))
        y = np.array(series)
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def _compute_trend_strength(self, series: List[float]) -> float:
        """Compute strength of trend (R-squared)"""
        if len(series) < 2:
            return 0.0
        
        x = np.arange(len(series))
        y = np.array(series)
        
        # Linear regression
        coeffs = np.polyfit(x, y, 1)
        y_pred = np.polyval(coeffs, x)
        
        # R-squared
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        return max(0.0, r_squared)
    
    def _detect_cycles(self, series: List[float], min_cycle_length: int = 3) -> List[Dict]:
        """Detect cyclical patterns in time series"""
        cycles = []
        
        if len(series) < min_cycle_length * 2:
            return cycles
        
        # Simple peak/valley detection
        peaks = []
        valleys = []
        
        for i in range(1, len(series) - 1):
            if series[i] > series[i-1] and series[i] > series[i+1]:
                peaks.append(i)
            elif series[i] < series[i-1] and series[i] < series[i+1]:
                valleys.append(i)
        
        # Analyze peak-to-peak distances
        if len(peaks) > 1:
            distances = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
            if distances:
                avg_cycle_length = np.mean(distances)
                cycle_regularity = 1.0 - (np.std(distances) / avg_cycle_length) if avg_cycle_length > 0 else 0.0
                
                cycles.append({
                    "type": "peak_cycle",
                    "average_length": avg_cycle_length,
                    "regularity": cycle_regularity,
                    "num_cycles": len(distances)
                })
        
        return cycles
    
    def _detect_critical_transitions(self, series: List[float], threshold: float = 0.1) -> List[Dict]:
        """Detect sudden transitions/phase changes"""
        transitions = []
        
        if len(series) < 3:
            return transitions
        
        # Compute derivatives (rate of change)
        derivatives = [series[i+1] - series[i] for i in range(len(series)-1)]
        
        # Find points where derivative changes significantly
        for i in range(1, len(derivatives)):
            change = abs(derivatives[i] - derivatives[i-1])
            if change > threshold:
                transitions.append({
                    "timestep": i,
                    "magnitude": change,
                    "direction": "increase" if derivatives[i] > derivatives[i-1] else "decrease"
                })
        
        return transitions
    
    def visualize_evolution(self, save_path: Optional[str] = None) -> None:
        """
        Create comprehensive visualization of thought evolution.
        """
        if not self.history:
            print("No evolution history to visualize. Run analyze_thought_evolution first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Thought Evolution Analysis - Phase 3', fontsize=16)
        
        timesteps = [snap.timestep for snap in self.history]
        coherence_scores = [snap.coherence_score for snap in self.history]
        diversity_scores = [snap.diversity_score for snap in self.history]
        alignment_scores = [snap.alignment_score for snap in self.history]
        
        # Plot 1: Coherence evolution
        axes[0, 0].plot(timesteps, coherence_scores, 'b-', linewidth=2, label='Coherence')
        axes[0, 0].set_title('Thought Coherence Evolution')
        axes[0, 0].set_xlabel('Diffusion Timestep')
        axes[0, 0].set_ylabel('Coherence Score')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: Diversity evolution
        axes[0, 1].plot(timesteps, diversity_scores, 'g-', linewidth=2, label='Diversity')
        axes[0, 1].set_title('Thought Diversity Evolution')
        axes[0, 1].set_xlabel('Diffusion Timestep')
        axes[0, 1].set_ylabel('Diversity Score')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3: Alignment evolution
        axes[1, 0].plot(timesteps, alignment_scores, 'r-', linewidth=2, label='Alignment')
        axes[1, 0].set_title('Thought-Text Alignment Evolution')
        axes[1, 0].set_xlabel('Diffusion Timestep')
        axes[1, 0].set_ylabel('Alignment Score')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 4: Combined metrics
        axes[1, 1].plot(timesteps, coherence_scores, 'b-', label='Coherence', alpha=0.7)
        axes[1, 1].plot(timesteps, diversity_scores, 'g-', label='Diversity', alpha=0.7)
        axes[1, 1].plot(timesteps, alignment_scores, 'r-', label='Alignment', alpha=0.7)
        axes[1, 1].set_title('All Metrics Combined')
        axes[1, 1].set_xlabel('Diffusion Timestep')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
    
    def export_analysis(self, filepath: str) -> None:
        """Export complete analysis results to JSON"""
        analysis_data = {
            "metadata": {
                "num_snapshots": len(self.history),
                "model_type": type(self.model).__name__,
                "device": str(self.device)
            },
            "snapshots": [],
            "patterns": self.detect_thought_patterns(),
            "summary_statistics": self._compute_summary_stats()
        }
        
        # Convert snapshots to serializable format
        for snap in self.history:
            snapshot_data = {
                "timestep": snap.timestep,
                "coherence_score": snap.coherence_score,
                "diversity_score": snap.diversity_score,
                "alignment_score": snap.alignment_score,
                "thought_tensor_stats": {
                    "shape": list(snap.thought_tensor.shape),
                    "mean": snap.thought_tensor.mean().item(),
                    "std": snap.thought_tensor.std().item(),
                    "min": snap.thought_tensor.min().item(),
                    "max": snap.thought_tensor.max().item()
                }
            }
            analysis_data["snapshots"].append(snapshot_data)
        
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"Analysis exported to {filepath}")
    
    def _compute_summary_stats(self) -> Dict:
        """Compute summary statistics across all snapshots"""
        if not self.history:
            return {}
        
        coherence_scores = [snap.coherence_score for snap in self.history]
        diversity_scores = [snap.diversity_score for snap in self.history]
        alignment_scores = [snap.alignment_score for snap in self.history]
        
        return {
            "coherence": {
                "mean": np.mean(coherence_scores),
                "std": np.std(coherence_scores),
                "min": np.min(coherence_scores),
                "max": np.max(coherence_scores)
            },
            "diversity": {
                "mean": np.mean(diversity_scores),
                "std": np.std(diversity_scores),
                "min": np.min(diversity_scores),
                "max": np.max(diversity_scores)
            },
            "alignment": {
                "mean": np.mean(alignment_scores),
                "std": np.std(alignment_scores),
                "min": np.min(alignment_scores),
                "max": np.max(alignment_scores)
            }
        }

def create_thought_heatmap(thought_tensor: torch.Tensor, title: str = "Thought Tensor Heatmap") -> None:
    """Create heatmap visualization of thought tensor"""
    # Convert to numpy and handle different tensor shapes
    if len(thought_tensor.shape) > 2:
        # Flatten or slice for visualization
        if thought_tensor.shape[0] == 1:  # Remove batch dimension
            tensor_2d = thought_tensor.squeeze(0)
        else:
            tensor_2d = thought_tensor[0]  # Take first batch item
        
        # If still 3D, take first slice
        if len(tensor_2d.shape) > 2:
            tensor_2d = tensor_2d[0]
    else:
        tensor_2d = thought_tensor
    
    tensor_np = tensor_2d.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(tensor_np, cmap='viridis', center=0)
    plt.title(title)
    plt.xlabel('Dimension')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()