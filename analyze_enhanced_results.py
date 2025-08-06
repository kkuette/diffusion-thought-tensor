#!/usr/bin/env python3
"""
Enhanced Results Analysis for DREAM Comparison Experiment
Processes comprehensive metrics including thought evolution and DREAM-specific analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import seaborn as sns
from dream_metrics_analyzer import compare_dream_effectiveness

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)

class EnhancedResultsAnalyzer:
    """Analyze comprehensive experimental results with enhanced DREAM metrics"""
    
    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        self.results = self._load_results()
        
    def _load_results(self) -> Dict:
        """Load experimental results"""
        with open(self.results_path / "final_results.json", 'r') as f:
            return json.load(f)
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        report = {
            'experiment_summary': self._analyze_experiment_summary(),
            'performance_comparison': self._analyze_performance_trends(),
            'dream_specific_analysis': self._analyze_dream_metrics(),
            'thought_evolution_analysis': self._analyze_thought_evolution(),
            'statistical_significance': self._compute_statistical_tests(),
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _analyze_experiment_summary(self) -> Dict:
        """Analyze overall experiment summary"""
        thought_metrics = self.results['thought_model']['eval_metrics']
        baseline_metrics = self.results['baseline_model']['eval_metrics']
        
        final_thought = thought_metrics[-1]
        final_baseline = baseline_metrics[-1]
        
        summary = {
            'total_epochs': len(thought_metrics),
            'final_thought_performance': {
                'loss': final_thought['loss'],
                'accuracy': final_thought['accuracy'],
                'perplexity': final_thought['perplexity']
            },
            'final_baseline_performance': {
                'loss': final_baseline['loss'],
                'accuracy': final_baseline['accuracy'],
                'perplexity': final_baseline['perplexity']
            },
            'overall_improvements': {
                'loss_improvement': (final_baseline['loss'] - final_thought['loss']) / final_baseline['loss'] * 100,
                'accuracy_improvement': (final_thought['accuracy'] - final_baseline['accuracy']) / final_baseline['accuracy'] * 100,
                'perplexity_improvement': (final_baseline['perplexity'] - final_thought['perplexity']) / final_baseline['perplexity'] * 100
            }
        }
        
        return summary
    
    def _analyze_performance_trends(self) -> Dict:
        """Analyze performance trends over epochs"""
        thought_losses = [m['loss'] for m in self.results['thought_model']['eval_metrics']]
        baseline_losses = [m['loss'] for m in self.results['baseline_model']['eval_metrics']]
        
        thought_accuracies = [m['accuracy'] for m in self.results['thought_model']['eval_metrics']]
        baseline_accuracies = [m['accuracy'] for m in self.results['baseline_model']['eval_metrics']]
        
        trends = {
            'loss_trends': {
                'thought_trend': self._compute_trend(thought_losses),
                'baseline_trend': self._compute_trend(baseline_losses),
                'convergence_analysis': self._analyze_convergence(thought_losses, baseline_losses)
            },
            'accuracy_trends': {
                'thought_trend': self._compute_trend(thought_accuracies),
                'baseline_trend': self._compute_trend(baseline_accuracies),
                'convergence_analysis': self._analyze_convergence(thought_accuracies, baseline_accuracies, higher_better=True)
            }
        }
        
        return trends
    
    def _analyze_dream_metrics(self) -> Dict:
        """Analyze DREAM-specific metrics"""
        thought_metrics = self.results['thought_model']['eval_metrics']
        baseline_metrics = self.results['baseline_model']['eval_metrics']
        
        dream_analysis = {}
        
        # Check if DREAM metrics are available
        if 'masked_accuracy' in thought_metrics[-1] and 'masked_accuracy' in baseline_metrics[-1]:
            final_thought = thought_metrics[-1]
            final_baseline = baseline_metrics[-1]
            
            # Use the comparison function from dream_metrics_analyzer
            comparison = compare_dream_effectiveness(final_thought, final_baseline)
            dream_analysis['effectiveness_comparison'] = comparison
            
            # Track DREAM metrics over time
            thought_masked_acc = [m.get('masked_accuracy', 0) for m in thought_metrics]
            baseline_masked_acc = [m.get('masked_accuracy', 0) for m in baseline_metrics]
            
            thought_unmasked_acc = [m.get('unmasked_accuracy', 0) for m in thought_metrics]
            baseline_unmasked_acc = [m.get('unmasked_accuracy', 0) for m in baseline_metrics]
            
            dream_analysis['temporal_analysis'] = {
                'masked_accuracy_trends': {
                    'thought_trend': self._compute_trend(thought_masked_acc),
                    'baseline_trend': self._compute_trend(baseline_masked_acc),
                    'final_difference': thought_masked_acc[-1] - baseline_masked_acc[-1]
                },
                'unmasked_accuracy_trends': {
                    'thought_trend': self._compute_trend(thought_unmasked_acc),
                    'baseline_trend': self._compute_trend(baseline_unmasked_acc),
                    'final_difference': thought_unmasked_acc[-1] - baseline_unmasked_acc[-1]
                },
                'bidirectional_effectiveness': {
                    'thought_final': final_thought.get('bidirectional_effectiveness', 1.0),
                    'baseline_final': final_baseline.get('bidirectional_effectiveness', 1.0),
                    'improvement': final_thought.get('bidirectional_effectiveness', 1.0) - final_baseline.get('bidirectional_effectiveness', 1.0)
                }
            }
        
        return dream_analysis
    
    def _analyze_thought_evolution(self) -> Dict:
        """Analyze thought evolution metrics if available"""
        thought_metrics = self.results['thought_model']['eval_metrics']
        
        evolution_analysis = {}
        
        # Check for thought-specific metrics
        thought_keys = [
            'thought_evolution_rate', 'thought_evolution_std', 'thought_stack_sparsity',
            'thought_stack_rms', 'thought_stack_dynamic_range', 'thought_evolution_consistency'
        ]
        
        available_metrics = {}
        for key in thought_keys:
            values = [m.get(key, None) for m in thought_metrics if key in m]
            if values and all(v is not None for v in values):
                available_metrics[key] = {
                    'values': values,
                    'trend': self._compute_trend(values),
                    'final_value': values[-1],
                    'change_from_start': values[-1] - values[0] if len(values) > 1 else 0
                }
        
        evolution_analysis['available_metrics'] = available_metrics
        
        # Analyze progressive unmasking effectiveness if available
        progressive_keys = [
            'progressive_progressive_accuracy_improvement',
            'progressive_unmasking_efficiency',
            'progressive_final_accuracy'
        ]
        
        progressive_metrics = {}
        for key in progressive_keys:
            values = [m.get(key, None) for m in thought_metrics if key in m]
            if values and all(v is not None for v in values):
                progressive_metrics[key] = {
                    'values': values,
                    'trend': self._compute_trend(values),
                    'final_value': values[-1]
                }
        
        if progressive_metrics:
            evolution_analysis['progressive_unmasking_analysis'] = progressive_metrics
        
        return evolution_analysis
    
    def _compute_statistical_tests(self) -> Dict:
        """Compute statistical significance tests"""
        thought_losses = [m['loss'] for m in self.results['thought_model']['eval_metrics']]
        baseline_losses = [m['loss'] for m in self.results['baseline_model']['eval_metrics']]
        
        thought_accuracies = [m['accuracy'] for m in self.results['thought_model']['eval_metrics']]
        baseline_accuracies = [m['accuracy'] for m in self.results['baseline_model']['eval_metrics']]
        
        try:
            from scipy import stats
            
            # Paired t-tests for loss and accuracy
            loss_ttest = stats.ttest_rel(baseline_losses, thought_losses)  # baseline - thought (improvement if positive)
            acc_ttest = stats.ttest_rel(thought_accuracies, baseline_accuracies)  # thought - baseline (improvement if positive)
            
            statistical_tests = {
                'loss_improvement': {
                    't_statistic': float(loss_ttest.statistic),
                    'p_value': float(loss_ttest.pvalue),
                    'significant': loss_ttest.pvalue < 0.05,
                    'mean_improvement': np.mean(np.array(baseline_losses) - np.array(thought_losses))
                },
                'accuracy_improvement': {
                    't_statistic': float(acc_ttest.statistic),
                    'p_value': float(acc_ttest.pvalue),
                    'significant': acc_ttest.pvalue < 0.05,
                    'mean_improvement': np.mean(np.array(thought_accuracies) - np.array(baseline_accuracies))
                }
            }
            
            # DREAM-specific statistical tests if available
            thought_metrics = self.results['thought_model']['eval_metrics']
            baseline_metrics = self.results['baseline_model']['eval_metrics']
            
            if 'masked_accuracy' in thought_metrics[-1]:
                thought_masked = [m.get('masked_accuracy', 0) for m in thought_metrics]
                baseline_masked = [m.get('masked_accuracy', 0) for m in baseline_metrics]
                
                masked_ttest = stats.ttest_rel(thought_masked, baseline_masked)
                statistical_tests['masked_accuracy_improvement'] = {
                    't_statistic': float(masked_ttest.statistic),
                    'p_value': float(masked_ttest.pvalue),
                    'significant': masked_ttest.pvalue < 0.05,
                    'mean_improvement': np.mean(np.array(thought_masked) - np.array(baseline_masked))
                }
            
        except ImportError:
            statistical_tests = {'error': 'scipy not available for statistical tests'}
        except Exception as e:
            statistical_tests = {'error': f'Error computing statistics: {str(e)}'}
        
        return statistical_tests
    
    def _generate_recommendations(self) -> Dict:
        """Generate actionable recommendations based on analysis"""
        recommendations = {
            'overall_assessment': '',
            'specific_recommendations': [],
            'next_steps': []
        }
        
        # Analyze final performance
        final_thought = self.results['thought_model']['eval_metrics'][-1]
        final_baseline = self.results['baseline_model']['eval_metrics'][-1]
        
        loss_improvement = (final_baseline['loss'] - final_thought['loss']) / final_baseline['loss'] * 100
        
        if loss_improvement > 5:
            recommendations['overall_assessment'] = "Strong evidence that thoughts significantly contribute to performance"
            recommendations['specific_recommendations'].extend([
                "Continue developing the thought system architecture",
                "Explore scaling up the thought system for larger models",
                "Investigate optimal thought stack depth and dimensions"
            ])
        elif loss_improvement > 2:
            recommendations['overall_assessment'] = "Moderate evidence of thought system benefits"
            recommendations['specific_recommendations'].extend([
                "Optimize thought system hyperparameters",
                "Experiment with different thought architectures",
                "Consider architectural improvements to increase benefits"
            ])
        elif loss_improvement > -2:
            recommendations['overall_assessment'] = "Inconclusive results - further investigation needed"
            recommendations['specific_recommendations'].extend([
                "Run longer training to see if benefits emerge",
                "Try different datasets or tasks",
                "Analyze thought utilization patterns"
            ])
        else:
            recommendations['overall_assessment'] = "Thoughts appear counterproductive in current implementation"
            recommendations['specific_recommendations'].extend([
                "Review thought system architecture for issues",
                "Consider simpler thought implementations",
                "Investigate why thoughts hurt performance"
            ])
        
        # Add DREAM-specific recommendations
        if 'masked_accuracy' in final_thought:
            masked_improvement = final_thought.get('masked_accuracy', 0) - final_baseline.get('masked_accuracy', 0)
            if masked_improvement > 0.02:
                recommendations['specific_recommendations'].append(
                    "Strong DREAM benefits - focus on masked token prediction improvements"
                )
            elif masked_improvement > 0:
                recommendations['specific_recommendations'].append(
                    "Moderate DREAM benefits - optimize for better masked prediction"
                )
            else:
                recommendations['specific_recommendations'].append(
                    "No DREAM-specific benefits - reconsider thought alignment with DREAM goals"
                )
        
        # Next steps
        recommendations['next_steps'] = [
            "Run statistical significance tests with more epochs",
            "Analyze thought activation patterns for insights",
            "Test on different datasets or tasks",
            "Experiment with thought system variants"
        ]
        
        return recommendations
    
    def _compute_trend(self, values: List[float]) -> Dict:
        """Compute trend analysis for a series of values"""
        if len(values) < 2:
            return {'trend': 'insufficient_data'}
        
        # Linear regression to determine trend
        x = np.arange(len(values))
        y = np.array(values)
        
        if len(values) == 2:
            slope, intercept = np.polyfit(x, y, 1)
            r_value, p_value, std_err = 0, 0, 0
        else:
            slope, intercept = np.polyfit(x, y, 1)
            correlation_matrix = np.corrcoef(x, y)
            r_value = correlation_matrix[0, 1] if correlation_matrix.shape == (2, 2) else 0
            p_value, std_err = 0, 0
        
        return {
            'slope': float(slope),
            'r_squared': float(r_value**2) if isinstance(r_value, (int, float)) else 0,
            'direction': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable',
            'strength': 'strong' if abs(slope) > np.std(values) * 0.1 else 'weak'
        }
    
    def _analyze_convergence(self, series1: List[float], series2: List[float], higher_better: bool = False) -> Dict:
        """Analyze convergence between two series"""
        if len(series1) != len(series2) or len(series1) < 2:
            return {'convergence': 'insufficient_data'}
        
        # Calculate differences over time
        differences = np.array(series1) - np.array(series2)
        if not higher_better:
            differences = -differences  # For loss, lower is better
        
        # Analyze trend in differences
        trend = self._compute_trend(differences.tolist())
        
        return {
            'final_difference': float(differences[-1]),
            'trend_in_difference': trend,
            'converging': trend['direction'] == 'declining',
            'diverging': trend['direction'] == 'improving'
        }
    
    def create_visualizations(self, output_dir: str):
        """Create comprehensive visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Performance comparison plots
        self._plot_performance_comparison(output_path)
        
        # DREAM-specific plots
        if 'masked_accuracy' in self.results['thought_model']['eval_metrics'][-1]:
            self._plot_dream_metrics(output_path)
        
        # Thought evolution plots if available
        self._plot_thought_evolution(output_path)
    
    def _plot_performance_comparison(self, output_path: Path):
        """Plot performance comparison over epochs"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Performance Comparison: Thought vs Baseline Models', fontsize=16)
        
        epochs = range(1, len(self.results['thought_model']['eval_metrics']) + 1)
        
        # Loss comparison
        thought_losses = [m['loss'] for m in self.results['thought_model']['eval_metrics']]
        baseline_losses = [m['loss'] for m in self.results['baseline_model']['eval_metrics']]
        
        axes[0, 0].plot(epochs, thought_losses, label='Thought Model', marker='o')
        axes[0, 0].plot(epochs, baseline_losses, label='Baseline Model', marker='s')
        axes[0, 0].set_title('Evaluation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy comparison
        thought_accs = [m['accuracy'] for m in self.results['thought_model']['eval_metrics']]
        baseline_accs = [m['accuracy'] for m in self.results['baseline_model']['eval_metrics']]
        
        axes[0, 1].plot(epochs, thought_accs, label='Thought Model', marker='o')
        axes[0, 1].plot(epochs, baseline_accs, label='Baseline Model', marker='s')
        axes[0, 1].set_title('Evaluation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Perplexity comparison
        thought_ppls = [m['perplexity'] for m in self.results['thought_model']['eval_metrics']]
        baseline_ppls = [m['perplexity'] for m in self.results['baseline_model']['eval_metrics']]
        
        axes[1, 0].plot(epochs, thought_ppls, label='Thought Model', marker='o')
        axes[1, 0].plot(epochs, baseline_ppls, label='Baseline Model', marker='s')
        axes[1, 0].set_title('Perplexity')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Perplexity')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance difference
        loss_diffs = np.array(baseline_losses) - np.array(thought_losses)
        acc_diffs = np.array(thought_accs) - np.array(baseline_accs)
        
        ax_diff = axes[1, 1]
        ax_diff.plot(epochs, loss_diffs, label='Loss Improvement', marker='o')
        ax_diff_acc = ax_diff.twinx()
        ax_diff_acc.plot(epochs, acc_diffs, label='Accuracy Improvement', marker='s', color='red')
        
        ax_diff.set_title('Performance Improvements (Thought vs Baseline)')
        ax_diff.set_xlabel('Epoch')
        ax_diff.set_ylabel('Loss Improvement', color='blue')
        ax_diff_acc.set_ylabel('Accuracy Improvement', color='red')
        ax_diff.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax_diff.get_legend_handles_labels()
        lines2, labels2 = ax_diff_acc.get_legend_handles_labels()
        ax_diff.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_path / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_dream_metrics(self, output_path: Path):
        """Plot DREAM-specific metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DREAM-Specific Metrics Analysis', fontsize=16)
        
        epochs = range(1, len(self.results['thought_model']['eval_metrics']) + 1)
        
        # Masked accuracy comparison
        thought_masked = [m.get('masked_accuracy', 0) for m in self.results['thought_model']['eval_metrics']]
        baseline_masked = [m.get('masked_accuracy', 0) for m in self.results['baseline_model']['eval_metrics']]
        
        axes[0, 0].plot(epochs, thought_masked, label='Thought Model', marker='o')
        axes[0, 0].plot(epochs, baseline_masked, label='Baseline Model', marker='s')
        axes[0, 0].set_title('Masked Token Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Unmasked accuracy comparison
        thought_unmasked = [m.get('unmasked_accuracy', 0) for m in self.results['thought_model']['eval_metrics']]
        baseline_unmasked = [m.get('unmasked_accuracy', 0) for m in self.results['baseline_model']['eval_metrics']]
        
        axes[0, 1].plot(epochs, thought_unmasked, label='Thought Model', marker='o')
        axes[0, 1].plot(epochs, baseline_unmasked, label='Baseline Model', marker='s')
        axes[0, 1].set_title('Unmasked Token Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bidirectional effectiveness
        thought_bidir = [m.get('bidirectional_effectiveness', 1) for m in self.results['thought_model']['eval_metrics']]
        baseline_bidir = [m.get('bidirectional_effectiveness', 1) for m in self.results['baseline_model']['eval_metrics']]
        
        axes[1, 0].plot(epochs, thought_bidir, label='Thought Model', marker='o')
        axes[1, 0].plot(epochs, baseline_bidir, label='Baseline Model', marker='s')
        axes[1, 0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect Balance')
        axes[1, 0].set_title('Bidirectional Effectiveness')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Masked/Unmasked Ratio')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # DREAM improvements
        masked_improvements = np.array(thought_masked) - np.array(baseline_masked)
        unmasked_improvements = np.array(thought_unmasked) - np.array(baseline_unmasked)
        
        axes[1, 1].plot(epochs, masked_improvements, label='Masked Improvement', marker='o')
        axes[1, 1].plot(epochs, unmasked_improvements, label='Unmasked Improvement', marker='s')
        axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('DREAM Accuracy Improvements')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy Improvement')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'dream_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_thought_evolution(self, output_path: Path):
        """Plot thought evolution metrics if available"""
        thought_metrics = self.results['thought_model']['eval_metrics']
        
        # Check what thought metrics are available
        available_keys = []
        for key in thought_metrics[0].keys():
            if key.startswith('thought_') and isinstance(thought_metrics[0][key], (int, float)):
                available_keys.append(key)
        
        if not available_keys:
            return  # No thought metrics to plot
        
        n_plots = len(available_keys)
        cols = 3
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        fig.suptitle('Thought Evolution Metrics', fontsize=16)
        
        # Always convert axes to a flattened array for consistent indexing
        if rows == 1 and cols == 1:
            axes = [axes]  # Single subplot case
        else:
            axes = np.array(axes).flatten()  # Multi-subplot case
        
        epochs = range(1, len(thought_metrics) + 1)
        
        for i, key in enumerate(available_keys):
            # Simple indexing - axes is always a 1D array now
            ax = axes[i]
            
            values = [m.get(key, 0) for m in thought_metrics]
            ax.plot(epochs, values, marker='o')
            ax.set_title(key.replace('_', ' ').title())
            ax.set_xlabel('Epoch')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_plots, rows * cols):
            if i < len(axes):
                axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'thought_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze enhanced experimental results")
    parser.add_argument("results_dir", help="Directory containing experimental results")
    parser.add_argument("--output", default="analysis_output", help="Output directory for analysis")
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = EnhancedResultsAnalyzer(args.results_dir)
    report = analyzer.generate_comprehensive_report()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Save comprehensive report
    with open(Path(args.output) / "comprehensive_analysis.json", 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    
    # Create visualizations
    analyzer.create_visualizations(args.output)
    
    # Print summary
    print("=" * 80)
    print("ENHANCED EXPERIMENTAL ANALYSIS SUMMARY")
    print("=" * 80)
    
    summary = report['experiment_summary']
    print(f"\nExperiment Overview:")
    print(f"  Total Epochs: {summary['total_epochs']}")
    print(f"  Final Loss Improvement: {summary['overall_improvements']['loss_improvement']:.2f}%")
    print(f"  Final Accuracy Improvement: {summary['overall_improvements']['accuracy_improvement']:.2f}%")
    
    if 'dream_specific_analysis' in report:
        dream = report['dream_specific_analysis']
        if 'effectiveness_comparison' in dream:
            comp = dream['effectiveness_comparison']
            print(f"\nDREAM Analysis:")
            print(f"  Interpretation: {comp.get('interpretation', 'N/A')}")
            print(f"  Recommendation: {comp.get('recommendation', 'N/A')}")
    
    print(f"\nRecommendations:")
    recs = report['recommendations']
    print(f"  Overall Assessment: {recs['overall_assessment']}")
    for rec in recs['specific_recommendations'][:3]:  # Show top 3
        print(f"  • {rec}")
    
    print(f"\n✅ Comprehensive analysis complete! Results saved to: {args.output}")

if __name__ == "__main__":
    main()