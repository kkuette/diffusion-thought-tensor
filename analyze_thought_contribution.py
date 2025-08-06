#!/usr/bin/env python3
"""
Analysis Script for Thought Contribution Comparison
Analyze results from controlled comparison experiment and perform additional tests

This script provides comprehensive analysis of whether thoughts contribute to performance,
including statistical testing, visualization, and interpretation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import argparse
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from diffusion_thought_tensor.model.stacked_3d_model import StackedDiffusionModel3D, MaskingStrategy
from diffusion_thought_tensor.model.baseline_dream_model import BaselineDreamModel
from diffusion_thought_tensor.data.huggingface_data_loader import create_huggingface_data_loaders


class ThoughtContributionAnalyzer:
    """Comprehensive analysis of thought system contribution"""
    
    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load experiment info
        with open(self.experiment_dir / "experiment_info.json") as f:
            self.experiment_info = json.load(f)
        
        # Load results
        with open(self.experiment_dir / "final_results.json") as f:
            self.results = json.load(f)
        
        print(f"📊 Analyzing experiment from: {experiment_dir}")
    
    def load_models(self, thought_model_path: str, baseline_model_path: str):
        """Load trained models for additional analysis"""
        # Load thought model
        self.thought_model = self._load_thought_model(thought_model_path)
        
        # Load baseline model  
        self.baseline_model = self._load_baseline_model(baseline_model_path)
        
        print(f"✅ Models loaded for analysis")
    
    def _load_thought_model(self, model_path: str):
        """Load thought model"""
        masking_strategy = MaskingStrategy(mask_ratio=0.7, confidence_threshold=0.75)
        
        model = StackedDiffusionModel3D(
            vocab_size=50257,
            embed_dim=720,
            num_layers=16,
            num_heads=16,
            thought_dims=(32, 32),
            stack_depth=16,
            thought_hidden_dim=256,
            max_seq_length=2048,
            dropout=0.1,
            masking_strategy=masking_strategy,
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def _load_baseline_model(self, model_path: str):
        """Load baseline model"""
        masking_strategy = MaskingStrategy(mask_ratio=0.7, confidence_threshold=0.75)
        
        model = BaselineDreamModel(
            vocab_size=50257,
            embed_dim=720,
            num_layers=16,
            num_heads=16,
            max_seq_length=2048,
            dropout=0.1,
            masking_strategy=masking_strategy,
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
    def analyze_training_curves(self):
        """Analyze and visualize training curves"""
        print("\n📈 Analyzing Training Curves...")
        
        # Extract data
        thought_train = self.results['thought_model']['train_losses']
        baseline_train = self.results['baseline_model']['train_losses']
        
        thought_eval = [m['loss'] for m in self.results['thought_model']['eval_metrics']]
        baseline_eval = [m['loss'] for m in self.results['baseline_model']['eval_metrics']]
        
        epochs = list(range(1, len(thought_train) + 1))
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Thought vs Baseline Model Comparison', fontsize=16)
        
        # Training loss
        ax1.plot(epochs, thought_train, 'b-', label='Thought Model', linewidth=2)
        ax1.plot(epochs, baseline_train, 'r-', label='Baseline Model', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss')
        ax1.set_title('Training Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Evaluation loss
        ax2.plot(epochs, thought_eval, 'b-', label='Thought Model', linewidth=2)
        ax2.plot(epochs, baseline_eval, 'r-', label='Baseline Model', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Evaluation Loss')
        ax2.set_title('Evaluation Loss Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Accuracy comparison
        thought_acc = [m['accuracy'] for m in self.results['thought_model']['eval_metrics']]
        baseline_acc = [m['accuracy'] for m in self.results['baseline_model']['eval_metrics']]
        
        ax3.plot(epochs, thought_acc, 'b-', label='Thought Model', linewidth=2)
        ax3.plot(epochs, baseline_acc, 'r-', label='Baseline Model', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Accuracy Curves')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Perplexity comparison
        thought_ppl = [m['perplexity'] for m in self.results['thought_model']['eval_metrics']]
        baseline_ppl = [m['perplexity'] for m in self.results['baseline_model']['eval_metrics']]
        
        ax4.plot(epochs, thought_ppl, 'b-', label='Thought Model', linewidth=2)
        ax4.plot(epochs, baseline_ppl, 'r-', label='Baseline Model', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Perplexity')
        ax4.set_title('Perplexity Curves')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.experiment_dir / "training_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"💾 Training curves saved to: {plot_path}")
        
        return fig
    
    def statistical_significance_test(self):
        """Perform statistical significance testing"""
        print("\n📊 Statistical Significance Analysis...")
        
        # Get final metrics
        thought_metrics = self.results['thought_model']['eval_metrics']
        baseline_metrics = self.results['baseline_model']['eval_metrics']
        
        # Extract loss values (treating epochs as samples)
        thought_losses = [m['loss'] for m in thought_metrics]
        baseline_losses = [m['loss'] for m in baseline_metrics]
        
        # Perform t-test
        t_stat, p_value = stats.ttest_rel(baseline_losses, thought_losses)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((np.std(thought_losses) ** 2) + (np.std(baseline_losses) ** 2)) / 2)
        cohens_d = (np.mean(baseline_losses) - np.mean(thought_losses)) / pooled_std
        
        print(f"📈 Paired t-test results:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Cohen's d (effect size): {cohens_d:.4f}")
        
        # Interpretation
        if p_value < 0.01:
            significance = "HIGHLY SIGNIFICANT (p < 0.01)"
        elif p_value < 0.05:
            significance = "SIGNIFICANT (p < 0.05)"
        elif p_value < 0.1:
            significance = "MARGINALLY SIGNIFICANT (p < 0.1)"
        else:
            significance = "NOT SIGNIFICANT (p >= 0.1)"
        
        print(f"  Significance: {significance}")
        
        if abs(cohens_d) > 0.8:
            effect_size = "LARGE EFFECT"
        elif abs(cohens_d) > 0.5:
            effect_size = "MEDIUM EFFECT"  
        elif abs(cohens_d) > 0.2:
            effect_size = "SMALL EFFECT"
        else:
            effect_size = "NEGLIGIBLE EFFECT"
        
        print(f"  Effect size: {effect_size}")
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significance': significance,
            'effect_size': effect_size
        }
    
    def ablation_study(self, test_data_loader):
        """Perform ablation study with loaded models"""
        if not hasattr(self, 'thought_model') or not hasattr(self, 'baseline_model'):
            print("❌ Models not loaded. Cannot perform ablation study.")
            return None
        
        print("\n🧪 Ablation Study...")
        
        results = {}
        
        with torch.no_grad():
            total_samples = 0
            
            # Initialize persistent stacks for different conditions
            batch_size = 2
            trained_stack = self.thought_model.initialize_stack(batch_size, self.device)
            zero_stack = torch.zeros_like(trained_stack)
            random_stack = torch.randn_like(trained_stack) * 0.1
            
            # Accumulate losses
            thought_loss = 0
            zero_thought_loss = 0
            random_thought_loss = 0
            baseline_loss = 0
            
            for batch in test_data_loader:
                if total_samples >= 1000:  # Limit for speed
                    break
                    
                tokens = batch['input_ids'][:batch_size].to(self.device)
                B, seq_len = tokens.shape
                
                timesteps = torch.randint(0, 500, (B,)).to(self.device)
                token_mask = torch.rand(B, seq_len) < 0.7
                token_mask = token_mask.to(self.device)
                
                # Adjust stacks to current batch size if needed
                if B != batch_size:
                    if B < batch_size:
                        current_trained = trained_stack[:B]
                        current_zero = zero_stack[:B]
                        current_random = random_stack[:B]
                    else:
                        # Skip if batch is larger to keep consistent
                        continue
                else:
                    current_trained = trained_stack
                    current_zero = zero_stack
                    current_random = random_stack
                
                # 1. Normal thought model with persistent stack evolution
                logits, _, updated_stack = self.thought_model(
                    tokens, current_trained, timesteps, token_mask, return_all_thoughts=True
                )
                trained_stack = updated_stack.detach()  # Update persistent stack
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    tokens[:, 1:].reshape(-1),
                    reduction='sum'
                )
                thought_loss += loss.item()
                
                # 2. Zero thought stack (no evolution)
                logits, _, _ = self.thought_model(tokens, current_zero, timesteps, token_mask)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    tokens[:, 1:].reshape(-1),
                    reduction='sum'
                )
                zero_thought_loss += loss.item()
                
                # 3. Random thought stack (evolves but from random start each time)
                fresh_random = torch.randn_like(current_random) * 0.1
                logits, _, _ = self.thought_model(tokens, fresh_random, timesteps, token_mask)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    tokens[:, 1:].reshape(-1),
                    reduction='sum'
                )
                random_thought_loss += loss.item()
                
                # 4. Baseline model
                logits = self.baseline_model(tokens, timesteps, token_mask)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    tokens[:, 1:].reshape(-1),
                    reduction='sum'
                )
                baseline_loss += loss.item()
                
                total_samples += B * (seq_len - 1)
        
        # Normalize losses
        thought_loss /= total_samples
        zero_thought_loss /= total_samples
        random_thought_loss /= total_samples
        baseline_loss /= total_samples
        
        results = {
            'trained_thoughts': thought_loss,
            'zero_thoughts': zero_thought_loss,
            'random_thoughts': random_thought_loss,
            'baseline_no_thoughts': baseline_loss
        }
        
        print(f"📊 Ablation Results (lower is better):")
        print(f"  Trained thoughts:     {thought_loss:.4f}")
        print(f"  Zero thoughts:        {zero_thought_loss:.4f}")
        print(f"  Random thoughts:      {random_thought_loss:.4f}")
        print(f"  Baseline (no system): {baseline_loss:.4f}")
        
        # Calculate improvements
        vs_zero = (zero_thought_loss - thought_loss) / zero_thought_loss * 100
        vs_random = (random_thought_loss - thought_loss) / random_thought_loss * 100
        vs_baseline = (baseline_loss - thought_loss) / baseline_loss * 100
        
        print(f"\n📈 Improvements of trained thoughts:")
        print(f"  vs Zero thoughts:     {vs_zero:+.2f}%")
        print(f"  vs Random thoughts:   {vs_random:+.2f}%")
        print(f"  vs Baseline:          {vs_baseline:+.2f}%")
        
        # Interpretation
        if vs_baseline > 5 and vs_zero > 5:
            print("✅ STRONG EVIDENCE: Trained thoughts significantly help")
        elif vs_baseline > 2:
            print("🟡 MODERATE EVIDENCE: Thoughts provide some benefit")
        elif vs_baseline > -2:
            print("⚖️ INCONCLUSIVE: No clear benefit from thoughts")
        else:
            print("❌ THOUGHTS ARE HARMFUL: Baseline is better")
        
        return results
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE THOUGHT CONTRIBUTION ANALYSIS")
        print("="*80)
        
        # Basic experiment info
        print(f"🔬 Experiment Details:")
        print(f"  Dataset: {self.experiment_info['dataset']}")
        print(f"  Training samples: {self.experiment_info['train_samples']:,}")
        print(f"  Evaluation samples: {self.experiment_info['eval_samples']:,}")
        print(f"  Thought model params: {self.experiment_info['thought_params'] / 1e6:.1f}M")
        print(f"  Baseline model params: {self.experiment_info['baseline_params'] / 1e6:.1f}M")
        
        # Final performance comparison
        final_thought = self.results['thought_model']['eval_metrics'][-1]
        final_baseline = self.results['baseline_model']['eval_metrics'][-1]
        
        print(f"\n📊 Final Performance:")
        print(f"{'Metric':<15} {'Thought':<12} {'Baseline':<12} {'Improvement':<12}")
        print("-" * 55)
        
        loss_imp = (final_baseline['loss'] - final_thought['loss']) / final_baseline['loss'] * 100
        acc_imp = (final_thought['accuracy'] - final_baseline['accuracy']) / final_baseline['accuracy'] * 100
        ppl_imp = (final_baseline['perplexity'] - final_thought['perplexity']) / final_baseline['perplexity'] * 100
        
        print(f"{'Loss':<15} {final_thought['loss']:<12.4f} {final_baseline['loss']:<12.4f} {loss_imp:<12.2f}%")
        print(f"{'Accuracy':<15} {final_thought['accuracy']:<12.4f} {final_baseline['accuracy']:<12.4f} {acc_imp:<12.2f}%")
        print(f"{'Perplexity':<15} {final_thought['perplexity']:<12.2f} {final_baseline['perplexity']:<12.2f} {ppl_imp:<12.2f}%")
        
        # Statistical analysis
        stats_results = self.statistical_significance_test()
        
        # Final conclusion
        print(f"\n🎯 FINAL CONCLUSION:")
        
        if loss_imp > 5 and stats_results['p_value'] < 0.05:
            conclusion = "✅ THOUGHTS SIGNIFICANTLY CONTRIBUTE TO PERFORMANCE"
            recommendation = "Continue developing and improving the thought system."
        elif loss_imp > 2:
            conclusion = "🟡 THOUGHTS PROVIDE MODEST IMPROVEMENTS"
            recommendation = "Thoughts help but may need optimization for better gains."
        elif loss_imp > -2:
            conclusion = "⚖️ NO CLEAR EVIDENCE OF THOUGHT CONTRIBUTION"
            recommendation = "Consider redesigning the thought system or focusing on other improvements."
        else:
            conclusion = "❌ THOUGHTS ARE COUNTERPRODUCTIVE"
            recommendation = "Remove thought system and focus on baseline improvements."
        
        print(f"  {conclusion}")
        print(f"  Recommendation: {recommendation}")
        
        # Parameter efficiency
        param_overhead = (self.experiment_info['thought_params'] - self.experiment_info['baseline_params']) / self.experiment_info['baseline_params'] * 100
        if loss_imp > 0:
            efficiency = loss_imp / param_overhead
            print(f"  Parameter efficiency: {efficiency:.3f}% improvement per 1% parameter increase")
        
        return {
            'final_performance': {
                'thought': final_thought,
                'baseline': final_baseline,
                'improvements': {'loss': loss_imp, 'accuracy': acc_imp, 'perplexity': ppl_imp}
            },
            'statistical_analysis': stats_results,
            'conclusion': conclusion,
            'recommendation': recommendation
        }


def main():
    parser = argparse.ArgumentParser(description="Analyze thought contribution experiment")
    parser.add_argument("--experiment_dir", required=True,
                       help="Directory containing experiment results")
    parser.add_argument("--thought_model", type=str,
                       help="Path to trained thought model (for ablation study)")
    parser.add_argument("--baseline_model", type=str,
                       help="Path to trained baseline model (for ablation study)")
    parser.add_argument("--dataset", default="roneneldan/TinyStories",
                       help="Dataset for additional testing")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ThoughtContributionAnalyzer(args.experiment_dir)
    
    # Analyze training curves
    analyzer.analyze_training_curves()
    
    # Load models for ablation if provided
    if args.thought_model and args.baseline_model:
        analyzer.load_models(args.thought_model, args.baseline_model)
        
        # Load test data
        _, test_loader = create_huggingface_data_loaders(
            dataset_name=args.dataset,
            train_samples=100,
            eval_samples=500,
            seq_length=512,
            batch_size=2
        )
        
        # Run ablation study
        analyzer.ablation_study(test_loader)
    
    # Generate final report
    report = analyzer.generate_final_report()
    
    # Save report
    report_path = Path(args.experiment_dir) / "analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Analysis complete! Report saved to: {report_path}")


if __name__ == "__main__":
    main()