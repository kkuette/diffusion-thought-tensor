#!/usr/bin/env python3
"""
Analyze thought quality from trained stacked 3D model checkpoints.
This script evaluates the spatial patterns, stack utilization, and evolution of thoughts.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import argparse

# Add the project root to Python path
import sys
sys.path.append(str(Path(__file__).parent))

from diffusion_thought_tensor.model.stacked_3d_model import StackedDiffusionModel3D, MaskingStrategy


class ThoughtQualityAnalyzer:
    """Analyze the quality and patterns of 3D thought stacks from trained models"""
    
    def __init__(self, model_path: str, config: Dict):
        self.model_path = Path(model_path)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
    def _load_model(self) -> StackedDiffusionModel3D:
        """Load the trained model"""
        # Create model with same config as training
        masking_strategy = MaskingStrategy(
            mask_ratio=0.7,
            confidence_threshold=0.75, 
            progressive_unmasking=True
        )
        
        model = StackedDiffusionModel3D(
            vocab_size=self.config.get('vocab_size', 50257),
            embed_dim=self.config.get('embed_dim', 768),
            num_layers=self.config.get('num_layers', 8),
            num_heads=self.config.get('num_heads', 12),
            thought_dims=tuple(self.config.get('thought_dims', [32, 32])),
            stack_depth=self.config.get('stack_depth', 16),
            thought_hidden_dim=self.config.get('thought_hidden_dim', 256),
            max_seq_length=self.config.get('max_seq_length', 2048),
            dropout=self.config.get('dropout', 0.1),
            masking_strategy=masking_strategy,
        ).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        print(f"✅ Loaded model from {self.model_path}")
        return model
    
    def analyze_thought_stack_patterns(self, batch_size: int = 4, seq_length: int = 128) -> Dict:
        """Analyze spatial and temporal patterns in the thought stack"""
        print("\n🧠 Analyzing Thought Stack Patterns...")
        
        with torch.no_grad():
            # Generate sample data
            tokens = torch.randint(0, self.config.get('vocab_size', 50257), 
                                 (batch_size, seq_length)).to(self.device)
            thought_stack = self.model.initialize_stack(batch_size, self.device)
            timesteps = torch.randint(0, 1000, (batch_size,)).to(self.device)
            
            # Forward pass
            logits, new_thought, updated_stack = self.model(
                tokens, thought_stack, timesteps, return_all_thoughts=True
            )
            
            # Analyze patterns
            results = {}
            
            # 1. Stack depth utilization
            stack_activity = self._analyze_stack_depth_utilization(updated_stack)
            results['stack_utilization'] = stack_activity
            
            # 2. Spatial patterns
            spatial_patterns = self._analyze_spatial_patterns(updated_stack)
            results['spatial_patterns'] = spatial_patterns
            
            # 3. Thought evolution smoothness  
            evolution_metrics = self._analyze_thought_evolution(thought_stack, updated_stack)
            results['evolution'] = evolution_metrics
            
            # 4. Information content
            info_metrics = self._analyze_information_content(updated_stack)
            results['information'] = info_metrics
            
        return results
    
    def _analyze_stack_depth_utilization(self, stack: torch.Tensor) -> Dict:
        """Analyze how different depths in the stack are utilized"""
        B, C, H, W, D = stack.shape
        
        # Compute activity per depth level
        depth_activity = []
        depth_variance = []
        depth_sparsity = []
        
        for d in range(D):
            depth_slice = stack[:, :, :, :, d]  # (B, C, H, W)
            
            # Activity metrics
            activity = torch.abs(depth_slice).mean().item()
            variance = depth_slice.var().item()
            sparsity = (torch.abs(depth_slice) < 0.01).float().mean().item()
            
            depth_activity.append(activity)
            depth_variance.append(variance)
            depth_sparsity.append(sparsity)
        
        return {
            'depth_activity': depth_activity,
            'depth_variance': depth_variance, 
            'depth_sparsity': depth_sparsity,
            'most_active_depth': int(np.argmax(depth_activity)),
            'least_active_depth': int(np.argmin(depth_activity)),
            'activity_balance': np.std(depth_activity) / np.mean(depth_activity)
        }
    
    def _analyze_spatial_patterns(self, stack: torch.Tensor) -> Dict:
        """Analyze spatial patterns within thoughts"""
        B, C, H, W, D = stack.shape
        
        # Average across batch and depth to get overall spatial pattern
        spatial_avg = stack.mean(dim=(0, 4))  # (C, H, W)
        
        # Compute spatial statistics
        center_activity = spatial_avg[:, H//2, W//2].abs().mean().item()
        edge_activity = torch.cat([
            spatial_avg[:, 0, :].flatten(),
            spatial_avg[:, -1, :].flatten(), 
            spatial_avg[:, :, 0].flatten(),
            spatial_avg[:, :, -1].flatten()
        ]).abs().mean().item()
        
        # Spatial gradients (measure of local structure)
        grad_h = torch.abs(spatial_avg[:, 1:, :] - spatial_avg[:, :-1, :]).mean().item()
        grad_w = torch.abs(spatial_avg[:, :, 1:] - spatial_avg[:, :, :-1]).mean().item()
        
        return {
            'center_vs_edge_ratio': center_activity / (edge_activity + 1e-8),
            'horizontal_gradient': grad_h,
            'vertical_gradient': grad_w,
            'spatial_coherence': 1.0 / (grad_h + grad_w + 1e-8),
            'spatial_sparsity': (spatial_avg.abs() < 0.01).float().mean().item()
        }
    
    def _analyze_thought_evolution(self, initial_stack: torch.Tensor, 
                                 updated_stack: torch.Tensor) -> Dict:
        """Analyze how thoughts evolve between initial and updated states"""
        
        # Compute per-depth changes
        depth_changes = []
        B, C, H, W, D = initial_stack.shape
        
        for d in range(D):
            initial = initial_stack[:, :, :, :, d]
            updated = updated_stack[:, :, :, :, d]
            change = torch.abs(updated - initial).mean().item()
            depth_changes.append(change)
        
        # Overall evolution metrics
        total_change = torch.abs(updated_stack - initial_stack).mean().item()
        relative_change = total_change / (initial_stack.abs().mean().item() + 1e-8)
        
        return {
            'total_change': total_change,
            'relative_change': relative_change,
            'depth_changes': depth_changes,
            'most_changed_depth': int(np.argmax(depth_changes)),
            'change_consistency': 1.0 - (np.std(depth_changes) / (np.mean(depth_changes) + 1e-8))
        }
    
    def _analyze_information_content(self, stack: torch.Tensor) -> Dict:
        """Analyze information content and redundancy in thoughts"""
        B, C, H, W, D = stack.shape
        
        # Flatten for easier analysis
        stack_flat = stack.view(B, -1, D)  # (B, C*H*W, D)
        
        # Compute correlations between depths
        correlations = []
        for b in range(B):
            corr_matrix = torch.corrcoef(stack_flat[b].T)  # (D, D)
            # Get upper triangular correlations (exclude diagonal)
            mask = torch.triu(torch.ones_like(corr_matrix), diagonal=1).bool()
            corr_values = corr_matrix[mask]
            correlations.extend(corr_values.tolist())
        
        avg_correlation = np.mean([abs(c) for c in correlations if not np.isnan(c)])
        
        # Compute entropy-like measure (diversity)
        stack_norm = torch.softmax(stack.flatten(0, 3), dim=-1)  # Normalize across depth
        entropy = -torch.sum(stack_norm * torch.log(stack_norm + 1e-8), dim=-1).mean().item()
        
        return {
            'inter_depth_correlation': avg_correlation,
            'depth_diversity': entropy,
            'information_redundancy': avg_correlation,  # Higher correlation = more redundancy
            'effective_depth_usage': D * (1.0 - avg_correlation)  # Estimate of unique depths
        }
    
    def visualize_results(self, results: Dict, save_path: Optional[str] = None):
        """Create visualizations of the analysis results"""
        print("\n📊 Creating Visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('3D Thought Stack Quality Analysis', fontsize=16)
        
        # 1. Stack depth utilization
        ax = axes[0, 0]
        depths = list(range(len(results['stack_utilization']['depth_activity'])))
        ax.bar(depths, results['stack_utilization']['depth_activity'])
        ax.set_title('Activity by Stack Depth')
        ax.set_xlabel('Depth Level')
        ax.set_ylabel('Average Activity')
        
        # 2. Depth variance
        ax = axes[0, 1]  
        ax.bar(depths, results['stack_utilization']['depth_variance'])
        ax.set_title('Variance by Stack Depth')
        ax.set_xlabel('Depth Level')
        ax.set_ylabel('Variance')
        
        # 3. Depth sparsity
        ax = axes[0, 2]
        ax.bar(depths, results['stack_utilization']['depth_sparsity'])
        ax.set_title('Sparsity by Stack Depth')
        ax.set_xlabel('Depth Level') 
        ax.set_ylabel('Sparsity (fraction near-zero)')
        
        # 4. Evolution changes by depth
        ax = axes[1, 0]
        ax.bar(depths, results['evolution']['depth_changes'])
        ax.set_title('Change by Stack Depth')
        ax.set_xlabel('Depth Level')
        ax.set_ylabel('Average Change')
        
        # 5. Spatial patterns summary
        ax = axes[1, 1]
        spatial_metrics = [
            results['spatial_patterns']['center_vs_edge_ratio'],
            results['spatial_patterns']['spatial_coherence'], 
            results['spatial_patterns']['spatial_sparsity'],
            results['information']['depth_diversity'],
            results['information']['effective_depth_usage'] / 16.0  # Normalize
        ]
        metric_names = ['Center/Edge', 'Coherence', 'Sparsity', 'Diversity', 'Eff. Depth']
        ax.bar(metric_names, spatial_metrics)
        ax.set_title('Spatial & Information Metrics')
        ax.tick_params(axis='x', rotation=45)
        
        # 6. Summary scores
        ax = axes[1, 2]
        
        # Compute overall quality scores
        utilization_score = 1.0 - results['stack_utilization']['activity_balance']
        evolution_score = results['evolution']['change_consistency'] 
        information_score = results['information']['depth_diversity'] / 10.0  # Normalize
        spatial_score = min(1.0, results['spatial_patterns']['spatial_coherence'] / 10.0)
        
        scores = [utilization_score, evolution_score, information_score, spatial_score]
        score_names = ['Stack\nUtilization', 'Evolution\nConsistency', 'Information\nDiversity', 'Spatial\nCoherence']
        
        bars = ax.bar(score_names, scores)
        ax.set_title('Overall Quality Scores')
        ax.set_ylabel('Score (0-1, higher is better)')
        ax.set_ylim(0, 1)
        
        # Color bars based on score
        for bar, score in zip(bars, scores):
            if score > 0.7:
                bar.set_color('green')
            elif score > 0.4:
                bar.set_color('orange') 
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved visualization to {save_path}")
        else:
            plt.show()
    
    def print_summary(self, results: Dict):
        """Print a summary of the analysis results"""
        print("\n" + "="*60)
        print("🎯 THOUGHT QUALITY ANALYSIS SUMMARY")
        print("="*60)
        
        # Stack utilization
        util = results['stack_utilization']
        print(f"\n📚 Stack Utilization:")
        print(f"  Most active depth: {util['most_active_depth']} (activity: {util['depth_activity'][util['most_active_depth']]:.3f})")
        print(f"  Least active depth: {util['least_active_depth']} (activity: {util['depth_activity'][util['least_active_depth']]:.3f})")
        print(f"  Activity balance: {util['activity_balance']:.3f} (lower is better)")
        
        # Spatial patterns
        spatial = results['spatial_patterns']
        print(f"\n🗺️  Spatial Patterns:")
        print(f"  Center vs edge activity: {spatial['center_vs_edge_ratio']:.3f}")
        print(f"  Spatial coherence: {spatial['spatial_coherence']:.3f}")
        print(f"  Spatial sparsity: {spatial['spatial_sparsity']:.3f}")
        
        # Evolution
        evo = results['evolution']
        print(f"\n🔄 Thought Evolution:")
        print(f"  Total change: {evo['total_change']:.3f}")
        print(f"  Relative change: {evo['relative_change']:.3f}")
        print(f"  Most changed depth: {evo['most_changed_depth']}")
        print(f"  Change consistency: {evo['change_consistency']:.3f}")
        
        # Information content
        info = results['information']
        print(f"\n💡 Information Content:")
        print(f"  Inter-depth correlation: {info['inter_depth_correlation']:.3f}")
        print(f"  Depth diversity: {info['depth_diversity']:.3f}")
        print(f"  Effective depth usage: {info['effective_depth_usage']:.1f}/{len(util['depth_activity'])}")
        
        # Overall assessment
        print(f"\n🏆 Overall Assessment:")
        if util['activity_balance'] < 0.3 and spatial['spatial_coherence'] > 5.0:
            print("  ✅ GOOD: Well-balanced stack with coherent spatial patterns")
        elif util['activity_balance'] < 0.5:
            print("  🟡 OKAY: Reasonable stack utilization, some areas for improvement")
        else:
            print("  🔴 POOR: Unbalanced stack usage, may need architecture adjustments")


def main():
    parser = argparse.ArgumentParser(description='Analyze thought quality from trained model')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, 
                       help='Path to model config JSON file')
    parser.add_argument('--save_viz', type=str,
                       help='Path to save visualization (optional)')
    
    args = parser.parse_args()
    
    # Default config based on dream_config.yaml
    default_config = {
        'vocab_size': 50257,
        'embed_dim': 720,
        'num_layers': 16, 
        'num_heads': 16,
        'thought_dims': [32, 32],
        'stack_depth': 16,
        'thought_hidden_dim': 720,
        'max_seq_length': 2048,
        'dropout': 0.1
    }
    
    # Load custom config if provided
    config = default_config
    if args.config_path:
        if args.config_path.endswith('.yaml') or args.config_path.endswith('.yml'):
            import yaml
            with open(args.config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            # Extract model config from YAML
            if 'model' in yaml_config:
                model_config = yaml_config['model']
                config.update({
                    'vocab_size': model_config.get('vocab_size', 50257),
                    'embed_dim': model_config.get('embed_dim', 720),
                    'num_layers': model_config.get('num_layers', 16),
                    'num_heads': model_config.get('num_heads', 16),
                    'max_seq_length': model_config.get('max_seq_length', 2048),
                    'dropout': model_config.get('dropout', 0.1)
                })
            if 'thought_tensor' in yaml_config:
                thought_config = yaml_config['thought_tensor']
                input_dims = thought_config.get('input_dims', [32, 32, 256])
                config.update({
                    'thought_dims': input_dims[:2],  # H, W
                    'stack_depth': thought_config.get('stack_size', 16),
                    'thought_hidden_dim': input_dims[2] if len(input_dims) > 2 else 720
                })
        else:
            with open(args.config_path, 'r') as f:
                custom_config = json.load(f)
            config.update(custom_config)
    
    # Run analysis
    analyzer = ThoughtQualityAnalyzer(args.model_path, config)
    results = analyzer.analyze_thought_stack_patterns()
    
    # Print results
    analyzer.print_summary(results)
    
    # Create visualizations
    analyzer.visualize_results(results, args.save_viz)
    
    print(f"\n✅ Analysis complete!")


if __name__ == "__main__":
    main()