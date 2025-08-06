#!/usr/bin/env python3
"""
Statistical Analysis for DREAM Comparison Experiment
Comprehensive statistical tests for thought vs baseline model comparison
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

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

try:
    from scipy import stats
    from scipy.stats import ttest_rel, ttest_ind, mannwhitneyu, wilcoxon
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Statistical tests will be limited.")

class StatisticalAnalyzer:
    """Perform comprehensive statistical analysis of experimental results"""
    
    def __init__(self, results_path: str):
        self.results_path = Path(results_path)
        self.results = self._load_results()
        
    def _load_results(self) -> Dict:
        """Load experimental results"""
        with open(self.results_path / "final_results.json", 'r') as f:
            return json.load(f)
    
    def run_comprehensive_analysis(self) -> Dict:
        """Run comprehensive statistical analysis"""
        analysis = {
            'basic_tests': self._run_basic_tests(),
            'dream_specific_tests': self._run_dream_tests(),
            'effect_size_analysis': self._compute_effect_sizes(),
            'confidence_intervals': self._compute_confidence_intervals(),
            'power_analysis': self._estimate_power(),
            'practical_significance': self._assess_practical_significance()
        }
        
        return analysis
    
    def _run_basic_tests(self) -> Dict:
        """Run basic statistical tests for core metrics"""
        if not SCIPY_AVAILABLE:
            return {'error': 'scipy not available'}
        
        thought_metrics = self.results['thought_model']['eval_metrics']
        baseline_metrics = self.results['baseline_model']['eval_metrics']
        
        # Extract time series
        thought_losses = np.array([m['loss'] for m in thought_metrics])
        baseline_losses = np.array([m['loss'] for m in baseline_metrics])
        
        thought_accuracies = np.array([m['accuracy'] for m in thought_metrics])
        baseline_accuracies = np.array([m['accuracy'] for m in baseline_metrics])
        
        thought_perplexities = np.array([m['perplexity'] for m in thought_metrics])
        baseline_perplexities = np.array([m['perplexity'] for m in baseline_metrics])
        
        basic_tests = {}
        
        # Paired t-tests (same epochs, different models)
        basic_tests['paired_tests'] = {
            'loss': self._paired_test_analysis(baseline_losses, thought_losses, 'lower_better'),
            'accuracy': self._paired_test_analysis(thought_accuracies, baseline_accuracies, 'higher_better'),
            'perplexity': self._paired_test_analysis(baseline_perplexities, thought_perplexities, 'lower_better')
        }
        
        # Final epoch comparison (independent samples)
        basic_tests['final_epoch_tests'] = {
            'loss_final': self._final_comparison(baseline_losses[-1], thought_losses[-1], 'lower_better'),
            'accuracy_final': self._final_comparison(thought_accuracies[-1], baseline_accuracies[-1], 'higher_better'),
            'perplexity_final': self._final_comparison(baseline_perplexities[-1], thought_perplexities[-1], 'lower_better')
        }
        
        # Trend analysis
        basic_tests['trend_analysis'] = {
            'thought_loss_trend': self._analyze_trend(thought_losses),
            'baseline_loss_trend': self._analyze_trend(baseline_losses),
            'thought_accuracy_trend': self._analyze_trend(thought_accuracies),
            'baseline_accuracy_trend': self._analyze_trend(baseline_accuracies)
        }
        
        return basic_tests
    
    def _run_dream_tests(self) -> Dict:
        """Run DREAM-specific statistical tests"""
        if not SCIPY_AVAILABLE:
            return {'error': 'scipy not available'}
        
        thought_metrics = self.results['thought_model']['eval_metrics']
        baseline_metrics = self.results['baseline_model']['eval_metrics']
        
        # Check if DREAM metrics are available
        if 'masked_accuracy' not in thought_metrics[-1] or 'masked_accuracy' not in baseline_metrics[-1]:
            return {'error': 'DREAM metrics not available'}
        
        # Extract DREAM time series
        thought_masked = np.array([m.get('masked_accuracy', 0) for m in thought_metrics])
        baseline_masked = np.array([m.get('masked_accuracy', 0) for m in baseline_metrics])
        
        thought_unmasked = np.array([m.get('unmasked_accuracy', 0) for m in thought_metrics])
        baseline_unmasked = np.array([m.get('unmasked_accuracy', 0) for m in baseline_metrics])
        
        thought_bidir = np.array([m.get('bidirectional_effectiveness', 1.0) for m in thought_metrics])
        baseline_bidir = np.array([m.get('bidirectional_effectiveness', 1.0) for m in baseline_metrics])
        
        dream_tests = {}
        
        # DREAM-specific paired tests
        dream_tests['paired_dream_tests'] = {
            'masked_accuracy': self._paired_test_analysis(thought_masked, baseline_masked, 'higher_better'),
            'unmasked_accuracy': self._paired_test_analysis(thought_unmasked, baseline_unmasked, 'higher_better'),
            'bidirectional_effectiveness': self._paired_test_analysis(thought_bidir, baseline_bidir, 'closer_to_one')
        }
        
        # DREAM specificity test: Are improvements specific to masked tokens?
        masked_improvement = thought_masked - baseline_masked
        unmasked_improvement = thought_unmasked - baseline_unmasked
        
        dream_specificity_test = self._paired_test_analysis(masked_improvement, unmasked_improvement, 'higher_better')
        dream_tests['dream_specificity'] = {
            **dream_specificity_test,
            'interpretation': self._interpret_dream_specificity(dream_specificity_test, 
                                                             masked_improvement[-1], unmasked_improvement[-1])
        }
        
        # Progressive improvement test: Do DREAM benefits increase over time?
        dream_tests['progressive_improvement'] = {
            'masked_improvement_trend': self._analyze_trend(masked_improvement),
            'unmasked_improvement_trend': self._analyze_trend(unmasked_improvement),
            'interpretation': self._interpret_progressive_improvement(masked_improvement, unmasked_improvement)
        }
        
        return dream_tests
    
    def _compute_effect_sizes(self) -> Dict:
        """Compute effect sizes for practical significance"""
        thought_metrics = self.results['thought_model']['eval_metrics']
        baseline_metrics = self.results['baseline_model']['eval_metrics']
        
        effect_sizes = {}
        
        # Cohen's d for main metrics
        metrics_to_analyze = [
            ('loss', 'lower_better'),
            ('accuracy', 'higher_better'),
            ('perplexity', 'lower_better')
        ]
        
        for metric_name, direction in metrics_to_analyze:
            thought_values = np.array([m[metric_name] for m in thought_metrics])
            baseline_values = np.array([m[metric_name] for m in baseline_metrics])
            
            effect_sizes[metric_name] = self._compute_cohens_d(
                thought_values, baseline_values, direction
            )
        
        # DREAM-specific effect sizes if available
        if 'masked_accuracy' in thought_metrics[-1]:
            dream_metrics = [
                ('masked_accuracy', 'higher_better'),
                ('unmasked_accuracy', 'higher_better'),
                ('bidirectional_effectiveness', 'closer_to_one')
            ]
            
            for metric_name, direction in dream_metrics:
                thought_values = np.array([m.get(metric_name, 0) for m in thought_metrics])
                baseline_values = np.array([m.get(metric_name, 0) for m in baseline_metrics])
                
                effect_sizes[f'dream_{metric_name}'] = self._compute_cohens_d(
                    thought_values, baseline_values, direction
                )
        
        return effect_sizes
    
    def _compute_confidence_intervals(self) -> Dict:
        """Compute confidence intervals for improvements"""
        if not SCIPY_AVAILABLE:
            return {'error': 'scipy not available'}
        
        thought_metrics = self.results['thought_model']['eval_metrics']
        baseline_metrics = self.results['baseline_model']['eval_metrics']
        
        confidence_intervals = {}
        
        # Main metrics
        metrics_data = {
            'loss_improvement': np.array([m['loss'] for m in baseline_metrics]) - np.array([m['loss'] for m in thought_metrics]),
            'accuracy_improvement': np.array([m['accuracy'] for m in thought_metrics]) - np.array([m['accuracy'] for m in baseline_metrics])
        }
        
        # DREAM metrics if available
        if 'masked_accuracy' in thought_metrics[-1]:
            metrics_data['masked_accuracy_improvement'] = (
                np.array([m.get('masked_accuracy', 0) for m in thought_metrics]) - 
                np.array([m.get('masked_accuracy', 0) for m in baseline_metrics])
            )
            
            metrics_data['unmasked_accuracy_improvement'] = (
                np.array([m.get('unmasked_accuracy', 0) for m in thought_metrics]) - 
                np.array([m.get('unmasked_accuracy', 0) for m in baseline_metrics])
            )
        
        for metric_name, improvements in metrics_data.items():
            confidence_intervals[metric_name] = self._compute_confidence_interval(improvements, confidence=0.95)
        
        return confidence_intervals
    
    def _estimate_power(self) -> Dict:
        """Estimate statistical power of the tests"""
        thought_metrics = self.results['thought_model']['eval_metrics']
        baseline_metrics = self.results['baseline_model']['eval_metrics']
        
        n_epochs = len(thought_metrics)
        
        power_analysis = {
            'sample_size': n_epochs,
            'power_estimates': {}
        }
        
        # Estimate power for main metrics based on observed effect sizes
        metrics_to_analyze = [
            ('loss', 'lower_better'),
            ('accuracy', 'higher_better')
        ]
        
        for metric_name, direction in metrics_to_analyze:
            thought_values = np.array([m[metric_name] for m in thought_metrics])
            baseline_values = np.array([m[metric_name] for m in baseline_metrics])
            
            effect_size = self._compute_cohens_d(thought_values, baseline_values, direction)['cohens_d']
            
            # Rough power estimation using effect size and sample size
            power_estimate = self._estimate_power_from_effect_size(effect_size, n_epochs)
            
            power_analysis['power_estimates'][metric_name] = {
                'estimated_power': power_estimate,
                'interpretation': self._interpret_power(power_estimate)
            }
        
        return power_analysis
    
    def _assess_practical_significance(self) -> Dict:
        """Assess practical significance of improvements"""
        thought_metrics = self.results['thought_model']['eval_metrics']
        baseline_metrics = self.results['baseline_model']['eval_metrics']
        
        final_thought = thought_metrics[-1]
        final_baseline = baseline_metrics[-1]
        
        practical_significance = {}
        
        # Define practical significance thresholds
        thresholds = {
            'loss': 0.05,  # 5% improvement
            'accuracy': 0.01,  # 1% improvement
            'perplexity': 10,  # 10 point improvement
            'masked_accuracy': 0.02,  # 2% improvement for DREAM
            'unmasked_accuracy': 0.01  # 1% improvement
        }
        
        # Assess main metrics
        loss_improvement = (final_baseline['loss'] - final_thought['loss']) / final_baseline['loss']
        accuracy_improvement = (final_thought['accuracy'] - final_baseline['accuracy']) / final_baseline['accuracy']
        perplexity_improvement = final_baseline['perplexity'] - final_thought['perplexity']
        
        practical_significance['main_metrics'] = {
            'loss': {
                'improvement': loss_improvement,
                'practically_significant': loss_improvement > thresholds['loss'],
                'threshold': thresholds['loss']
            },
            'accuracy': {
                'improvement': accuracy_improvement,
                'practically_significant': accuracy_improvement > thresholds['accuracy'],
                'threshold': thresholds['accuracy']
            },
            'perplexity': {
                'improvement': perplexity_improvement,
                'practically_significant': perplexity_improvement > thresholds['perplexity'],
                'threshold': thresholds['perplexity']
            }
        }
        
        # Assess DREAM metrics if available
        if 'masked_accuracy' in final_thought:
            masked_improvement = final_thought.get('masked_accuracy', 0) - final_baseline.get('masked_accuracy', 0)
            unmasked_improvement = final_thought.get('unmasked_accuracy', 0) - final_baseline.get('unmasked_accuracy', 0)
            
            practical_significance['dream_metrics'] = {
                'masked_accuracy': {
                    'improvement': masked_improvement,
                    'practically_significant': masked_improvement > thresholds['masked_accuracy'],
                    'threshold': thresholds['masked_accuracy']
                },
                'unmasked_accuracy': {
                    'improvement': unmasked_improvement,
                    'practically_significant': unmasked_improvement > thresholds['unmasked_accuracy'],
                    'threshold': thresholds['unmasked_accuracy']
                },
                'dream_specificity': {
                    'masked_better_than_unmasked': masked_improvement > unmasked_improvement,
                    'difference': masked_improvement - unmasked_improvement
                }
            }
        
        return practical_significance
    
    def _paired_test_analysis(self, series1: np.ndarray, series2: np.ndarray, direction: str) -> Dict:
        """Perform comprehensive paired test analysis"""
        if not SCIPY_AVAILABLE:
            return {'error': 'scipy not available'}
        
        # Paired t-test
        if direction == 'lower_better':
            # series1 should be lower than series2
            t_stat, p_value = ttest_rel(series2, series1)
            mean_improvement = np.mean(series2 - series1)
        elif direction == 'higher_better':
            # series1 should be higher than series2
            t_stat, p_value = ttest_rel(series1, series2)
            mean_improvement = np.mean(series1 - series2)
        else:  # closer_to_one
            # Analyze deviation from 1.0
            diff1 = np.abs(series1 - 1.0)
            diff2 = np.abs(series2 - 1.0)
            t_stat, p_value = ttest_rel(diff2, diff1)
            mean_improvement = np.mean(diff2 - diff1)
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        try:
            if direction == 'lower_better':
                wilcoxon_stat, wilcoxon_p = wilcoxon(series2, series1, alternative='greater')
            elif direction == 'higher_better':
                wilcoxon_stat, wilcoxon_p = wilcoxon(series1, series2, alternative='greater')
            else:
                diff1 = np.abs(series1 - 1.0)
                diff2 = np.abs(series2 - 1.0)
                wilcoxon_stat, wilcoxon_p = wilcoxon(diff2, diff1, alternative='greater')
        except Exception as e:
            wilcoxon_stat, wilcoxon_p = None, None
        
        return {
            'paired_t_test': {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant_05': p_value < 0.05,
                'significant_01': p_value < 0.01
            },
            'wilcoxon_test': {
                'statistic': float(wilcoxon_stat) if wilcoxon_stat is not None else None,
                'p_value': float(wilcoxon_p) if wilcoxon_p is not None else None,
                'significant_05': wilcoxon_p < 0.05 if wilcoxon_p is not None else None
            },
            'mean_improvement': float(mean_improvement),
            'interpretation': self._interpret_test_result(p_value, mean_improvement, direction)
        }
    
    def _final_comparison(self, value1: float, value2: float, direction: str) -> Dict:
        """Compare final epoch values"""
        if direction == 'lower_better':
            improvement = value2 - value1  # value1 should be lower
            better = value1 < value2
        elif direction == 'higher_better':
            improvement = value1 - value2  # value1 should be higher
            better = value1 > value2
        else:  # closer_to_one
            improvement = abs(value2 - 1.0) - abs(value1 - 1.0)  # value1 should be closer to 1
            better = abs(value1 - 1.0) < abs(value2 - 1.0)
        
        return {
            'value1': float(value1),
            'value2': float(value2),
            'improvement': float(improvement),
            'value1_better': better,
            'interpretation': 'improvement' if better else 'regression'
        }
    
    def _analyze_trend(self, series: np.ndarray) -> Dict:
        """Analyze trend in a time series"""
        if len(series) < 3:
            return {'error': 'insufficient data'}
        
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
        
        return {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_value': float(r_value),
            'r_squared': float(r_value**2),
            'p_value': float(p_value),
            'std_err': float(std_err),
            'significant_trend': p_value < 0.05,
            'direction': 'improving' if slope > 0 else 'declining',
            'strength': 'strong' if abs(r_value) > 0.7 else 'moderate' if abs(r_value) > 0.3 else 'weak'
        }
    
    def _compute_cohens_d(self, series1: np.ndarray, series2: np.ndarray, direction: str) -> Dict:
        """Compute Cohen's d effect size"""
        if direction == 'lower_better':
            # series1 should be lower than series2
            mean_diff = np.mean(series2) - np.mean(series1)
        elif direction == 'higher_better':
            # series1 should be higher than series2
            mean_diff = np.mean(series1) - np.mean(series2)
        else:  # closer_to_one
            mean_diff = np.mean(np.abs(series2 - 1.0)) - np.mean(np.abs(series1 - 1.0))
        
        pooled_std = np.sqrt((np.var(series1, ddof=1) + np.var(series2, ddof=1)) / 2)
        
        if pooled_std == 0:
            cohens_d = 0
        else:
            cohens_d = mean_diff / pooled_std
        
        return {
            'cohens_d': float(cohens_d),
            'interpretation': self._interpret_effect_size(abs(cohens_d)),
            'mean_difference': float(mean_diff),
            'pooled_std': float(pooled_std)
        }
    
    def _compute_confidence_interval(self, data: np.ndarray, confidence: float = 0.95) -> Dict:
        """Compute confidence interval for mean"""
        if not SCIPY_AVAILABLE:
            return {'error': 'scipy not available'}
        
        n = len(data)
        mean = np.mean(data)
        std_err = stats.sem(data)
        
        # t-distribution for small samples
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin_error = t_val * std_err
        
        return {
            'mean': float(mean),
            'std_error': float(std_err),
            'confidence_level': confidence,
            'lower_bound': float(mean - margin_error),
            'upper_bound': float(mean + margin_error),
            'margin_error': float(margin_error),
            'includes_zero': (mean - margin_error) <= 0 <= (mean + margin_error)
        }
    
    def _estimate_power_from_effect_size(self, effect_size: float, n: int) -> float:
        """Rough power estimation based on effect size and sample size"""
        # This is a simplified estimation - for precise power analysis, use proper tools
        if abs(effect_size) < 0.2:
            base_power = 0.2
        elif abs(effect_size) < 0.5:
            base_power = 0.5
        elif abs(effect_size) < 0.8:
            base_power = 0.8
        else:
            base_power = 0.95
        
        # Adjust for sample size
        if n < 5:
            power_adjustment = 0.5
        elif n < 10:
            power_adjustment = 0.7
        elif n < 20:
            power_adjustment = 0.9
        else:
            power_adjustment = 1.0
        
        return min(base_power * power_adjustment, 0.99)
    
    def _interpret_test_result(self, p_value: float, improvement: float, direction: str) -> str:
        """Interpret statistical test results"""
        if p_value < 0.01:
            significance = "highly significant"
        elif p_value < 0.05:
            significance = "significant"
        elif p_value < 0.10:
            significance = "marginally significant"
        else:
            significance = "not significant"
        
        if improvement > 0:
            direction_text = "improvement"
        elif improvement < 0:
            direction_text = "regression"
        else:
            direction_text = "no change"
        
        return f"{significance} {direction_text} (p={p_value:.4f})"
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        if cohens_d < 0.2:
            return "negligible"
        elif cohens_d < 0.5:
            return "small"
        elif cohens_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_power(self, power: float) -> str:
        """Interpret statistical power"""
        if power < 0.5:
            return "low power - high risk of missing real effects"
        elif power < 0.8:
            return "moderate power - some risk of missing effects"
        else:
            return "adequate power - good chance of detecting real effects"
    
    def _interpret_dream_specificity(self, test_result: Dict, masked_improvement: float, unmasked_improvement: float) -> str:
        """Interpret DREAM specificity test results"""
        if test_result['paired_t_test']['significant_05']:
            if masked_improvement > unmasked_improvement:
                return "Thoughts significantly benefit masked tokens more than unmasked (DREAM-specific benefit)"
            else:
                return "Thoughts significantly benefit unmasked tokens more than masked (not DREAM-specific)"
        else:
            return "No significant difference in benefit between masked and unmasked tokens"
    
    def _interpret_progressive_improvement(self, masked_improvements: np.ndarray, unmasked_improvements: np.ndarray) -> str:
        """Interpret progressive improvement patterns"""
        masked_trend = self._analyze_trend(masked_improvements)
        unmasked_trend = self._analyze_trend(unmasked_improvements)
        
        interpretations = []
        
        if masked_trend['significant_trend']:
            if masked_trend['direction'] == 'improving':
                interpretations.append("DREAM masked benefits increase significantly over time")
            else:
                interpretations.append("DREAM masked benefits decrease significantly over time")
        
        if unmasked_trend['significant_trend']:
            if unmasked_trend['direction'] == 'improving':
                interpretations.append("Unmasked benefits increase significantly over time")
            else:
                interpretations.append("Unmasked benefits decrease significantly over time")
        
        if not interpretations:
            interpretations.append("No significant trends in DREAM benefit progression")
        
        return "; ".join(interpretations)


def main():
    """Main statistical analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Statistical analysis of experimental results")
    parser.add_argument("results_dir", help="Directory containing experimental results")
    parser.add_argument("--output", default="statistical_analysis.json", help="Output file for analysis")
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = StatisticalAnalyzer(args.results_dir)
    analysis = analyzer.run_comprehensive_analysis()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(analysis, f, indent=2, cls=NumpyEncoder)
    
    # Print summary
    print("=" * 80)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("=" * 80)
    
    if 'basic_tests' in analysis and 'paired_tests' in analysis['basic_tests']:
        basic = analysis['basic_tests']['paired_tests']
        
        print(f"\nCore Metrics (Paired Tests):")
        for metric, test in basic.items():
            p_val = test['paired_t_test']['p_value']
            improvement = test['mean_improvement']
            print(f"  {metric.title()}: {test['interpretation']}")
    
    if 'dream_specific_tests' in analysis and 'paired_dream_tests' in analysis['dream_specific_tests']:
        dream = analysis['dream_specific_tests']['paired_dream_tests']
        
        print(f"\nDREAM-Specific Metrics:")
        for metric, test in dream.items():
            print(f"  {metric.replace('_', ' ').title()}: {test['interpretation']}")
    
    if 'effect_size_analysis' in analysis:
        print(f"\nEffect Sizes:")
        for metric, effect in analysis['effect_size_analysis'].items():
            print(f"  {metric.title()}: {effect['interpretation']} (d={effect['cohens_d']:.3f})")
    
    if 'practical_significance' in analysis:
        practical = analysis['practical_significance']
        
        print(f"\nPractical Significance:")
        for category, metrics in practical.items():
            if isinstance(metrics, dict):
                for metric, result in metrics.items():
                    if isinstance(result, dict) and 'practically_significant' in result:
                        status = "✅" if result['practically_significant'] else "❌"
                        print(f"  {metric.title()}: {status} ({result['improvement']:.4f})")
    
    print(f"\n✅ Statistical analysis complete! Results saved to: {args.output}")


if __name__ == "__main__":
    main()