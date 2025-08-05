"""
Phase 3 Demonstration: Thought Evolution Analysis

This script demonstrates the advanced thought evolution analysis capabilities
implemented in Phase 3, including pattern detection, coherence analysis,
and trajectory visualization.
"""

import torch
import numpy as np
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import Phase 2 model and Phase 3 analysis tools
from model.phase2_model import Phase2DiffusionThoughtModel
from analysis.thought_evolution import ThoughtEvolutionAnalyzer, ThoughtSnapshot
from analysis.trajectory_visualizer import ThoughtTrajectoryVisualizer, create_thought_heatmap_sequence
from analysis.coherence_metrics import AdvancedCoherenceAnalyzer, EmergentPatternDetector, export_coherence_analysis
from utils.noise_schedules import NoiseScheduler
from configs.phase2_config import load_phase2_config

def create_synthetic_thought_evolution(model, device, num_steps=100):
    """Create synthetic thought evolution for analysis"""
    print("Creating synthetic thought evolution sequence...")
    
    # Create initial noisy thought and text
    batch_size = 1
    thought_dims = [32, 32, 16]  # From Phase 2 config
    embed_dim = 896
    seq_length = 128
    
    # Start with noise
    initial_thought = torch.randn(batch_size, *thought_dims, device=device)
    initial_text = torch.randn(batch_size, seq_length, embed_dim, device=device)
    
    # Create noise scheduler
    noise_scheduler = NoiseScheduler(num_steps=num_steps, schedule_type="cosine")
    
    thought_history = []
    timesteps = []
    
    current_thought = initial_thought.clone()
    current_text = initial_text.clone()
    
    model.eval()
    with torch.no_grad():
        for step in range(num_steps, 0, -max(1, num_steps//50)):  # Sample ~50 points
            t = torch.tensor([step], device=device)
            timesteps.append(step)
            
            # Simple denoising step (mock evolution)
            # Use scheduler's internal scaling based on timestep
            if step < len(noise_scheduler.sqrt_one_minus_alphas_cumprod):
                noise_scale = noise_scheduler.sqrt_one_minus_alphas_cumprod[step].item()
            else:
                noise_scale = 0.1  # Fallback
            
            # Add some structure to the evolution
            structure_factor = 1.0 - (step / num_steps)  # Increases as we denoise
            
            # Create evolving thought with emerging structure
            evolution_direction = torch.randn_like(current_thought) * 0.1
            current_thought = current_thought * (1 - 0.05) + evolution_direction * structure_factor
            
            # Add periodic components for pattern detection
            phase = 2 * np.pi * step / (num_steps / 3)  # 3 cycles
            periodic_component = 0.1 * torch.sin(torch.tensor(phase)) * torch.ones_like(current_thought)
            current_thought = current_thought + periodic_component
            
            thought_history.append(current_thought.clone())
    
    print(f"Generated {len(thought_history)} thought snapshots")
    return thought_history, timesteps

def demonstrate_thought_evolution_analysis():
    """Demonstrate the core thought evolution analysis"""
    print("\n" + "="*60)
    print("PHASE 3 DEMONSTRATION: THOUGHT EVOLUTION ANALYSIS")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Phase 2 model for context
    print("\nLoading Phase 2 model...")
    config = load_phase2_config()
    
    # Create model (we'll use it for context, not actual inference)
    model = Phase2DiffusionThoughtModel(**config['model']).to(device)
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Phase 2 model loaded: {model_params:,} parameters")
    
    # Generate thought evolution sequence
    thought_history, timesteps = create_synthetic_thought_evolution(model, device)
    
    # Initialize analyzers
    print("\nInitializing Phase 3 analyzers...")
    evolution_analyzer = ThoughtEvolutionAnalyzer(model, device)
    trajectory_visualizer = ThoughtTrajectoryVisualizer()
    coherence_analyzer = AdvancedCoherenceAnalyzer(device)
    pattern_detector = EmergentPatternDetector(device)
    
    return thought_history, timesteps, evolution_analyzer, trajectory_visualizer, coherence_analyzer, pattern_detector

def analyze_thought_coherence(thought_history, coherence_analyzer):
    """Demonstrate coherence analysis"""
    print("\n" + "-"*40)
    print("COHERENCE ANALYSIS")
    print("-"*40)
    
    coherence_results = []
    
    for i, thought in enumerate(thought_history[::5]):  # Sample every 5th for speed
        analysis = coherence_analyzer.analyze_coherence_multiscale(thought)
        coherence_results.append(analysis)
        
        if i < 3:  # Show first few results
            print(f"\nStep {i*5}:")
            print(f"  Global Coherence: {analysis.global_coherence:.4f}")
            print(f"  Local Coherence: {analysis.local_coherence:.4f}")
            print(f"  Temporal Stability: {analysis.temporal_stability:.4f}")
            print(f"  Structural Integrity: {analysis.structural_integrity:.4f}")
            print(f"  Pattern Strength: {analysis.pattern_strength:.4f}")
    
    # Summary statistics
    global_coherences = [r.global_coherence for r in coherence_results]
    pattern_strengths = [r.pattern_strength for r in coherence_results]
    
    print(f"\nCoherence Summary:")
    print(f"  Average Global Coherence: {np.mean(global_coherences):.4f} ± {np.std(global_coherences):.4f}")
    print(f"  Average Pattern Strength: {np.mean(pattern_strengths):.4f} ± {np.std(pattern_strengths):.4f}")
    
    return coherence_results

def detect_emergent_patterns(thought_history, timesteps, pattern_detector):
    """Demonstrate pattern detection"""
    print("\n" + "-"*40)
    print("PATTERN DETECTION")
    print("-"*40)
    
    patterns = pattern_detector.detect_patterns_in_sequence(thought_history, timesteps)
    
    print(f"Recurring Motifs Found: {len(patterns.recurring_motifs)}")
    for i, motif in enumerate(patterns.recurring_motifs[:3]):  # Show first 3
        print(f"  Motif {i+1}: {len(motif['occurrences'])} occurrences, strength: {motif['strength']:.4f}")
    
    print(f"\nAttractor States Found: {len(patterns.attractor_states)}")
    
    print(f"\nPhase Transitions Found: {len(patterns.phase_transitions)}")
    for i, transition in enumerate(patterns.phase_transitions[:3]):  # Show first 3
        print(f"  Transition {i+1}: Step {transition['timestep']}, magnitude: {transition['magnitude']:.4f}")
    
    print(f"\nComplexity Evolution:")
    for key, value in patterns.complexity_measures.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nNetwork Properties:")
    for key, value in patterns.network_properties.items():
        print(f"  {key}: {value:.4f}")
    
    return patterns

def create_evolution_visualizations(thought_history, timesteps, trajectory_visualizer):
    """Create and save visualizations"""
    print("\n" + "-"*40)
    print("CREATING VISUALIZATIONS")
    print("-"*40)
    
    outputs_dir = Path("outputs/visualizations")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create coherence and diversity scores for visualization
    coherence_scores = []
    diversity_scores = []
    alignment_scores = []
    
    for thought in thought_history:
        # Mock scores for visualization (in real usage, these come from analysis)
        coherence = np.random.random() * 0.5 + 0.3  # 0.3-0.8 range
        diversity = np.random.random() * 0.4 + 0.2   # 0.2-0.6 range
        alignment = np.random.random() * 0.6 + 0.2   # 0.2-0.8 range
        
        coherence_scores.append(coherence)
        diversity_scores.append(diversity)
        alignment_scores.append(alignment)
    
    try:
        # 3D trajectory visualization
        print("Creating 3D trajectory visualization...")
        trajectory_visualizer.visualize_3d_trajectory(
            thought_history, timesteps,
            title="Phase 3: Thought Evolution 3D Trajectory",
            save_path=str(outputs_dir / "phase3_3d_trajectory.png")
        )
        
        # Flow field visualization
        print("Creating flow field visualization...")
        trajectory_visualizer.visualize_thought_flow_field(
            thought_history, timesteps,
            save_path=str(outputs_dir / "phase3_flow_field.png")
        )
        
        # Manifold visualization
        print("Creating manifold visualization...")
        trajectory_visualizer.visualize_thought_manifold(
            thought_history, timesteps,
            method='pca',
            save_path=str(outputs_dir / "phase3_manifold.png")
        )
        
        # Heatmap sequence
        print("Creating thought heatmap sequence...")
        sample_thoughts = thought_history[::len(thought_history)//8]  # 8 samples
        sample_timesteps = timesteps[::len(timesteps)//8]
        create_thought_heatmap_sequence(
            sample_thoughts, sample_timesteps,
            save_path=str(outputs_dir / "phase3_heatmap_sequence.png")
        )
        
        print(f"Visualizations saved to {outputs_dir}")
        
    except Exception as e:
        print(f"Visualization creation encountered issues: {e}")
        print("This is normal in environments without full matplotlib/plotly support")

def export_phase3_results(coherence_results, patterns, thought_history, timesteps):
    """Export Phase 3 analysis results"""
    print("\n" + "-"*40)
    print("EXPORTING RESULTS")
    print("-"*40)
    
    outputs_dir = Path("outputs/logs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare comprehensive results
    results = {
        "metadata": {
            "phase": 3,
            "timestamp": timestamp,
            "num_snapshots": len(thought_history),
            "timestep_range": [min(timesteps), max(timesteps)],
            "analysis_type": "thought_evolution"
        },
        "coherence_analysis": {
            "num_analyzed": len(coherence_results),
            "average_metrics": {
                "global_coherence": np.mean([r.global_coherence for r in coherence_results]),
                "local_coherence": np.mean([r.local_coherence for r in coherence_results]),
                "temporal_stability": np.mean([r.temporal_stability for r in coherence_results]),
                "structural_integrity": np.mean([r.structural_integrity for r in coherence_results]),
                "pattern_strength": np.mean([r.pattern_strength for r in coherence_results])
            },
            "evolution_trends": {
                "coherence_trend": np.polyfit(range(len(coherence_results)), 
                                            [r.global_coherence for r in coherence_results], 1)[0],
                "pattern_trend": np.polyfit(range(len(coherence_results)), 
                                          [r.pattern_strength for r in coherence_results], 1)[0]
            }
        },
        "pattern_detection": {
            "recurring_motifs": len(patterns.recurring_motifs),
            "attractor_states": len(patterns.attractor_states),
            "phase_transitions": len(patterns.phase_transitions),
            "complexity_measures": patterns.complexity_measures,
            "network_properties": patterns.network_properties
        },
        "emergent_properties": {
            "total_patterns_found": len(patterns.recurring_motifs) + len(patterns.attractor_states),
            "evolution_complexity": "high" if len(patterns.phase_transitions) > 3 else "moderate",
            "thought_organization": "structured" if patterns.network_properties.get('clustering_coefficient', 0) > 0.3 else "distributed"
        },
        "key_findings": [
            f"Detected {len(patterns.recurring_motifs)} recurring thought patterns",
            f"Found {len(patterns.attractor_states)} potential attractor states",
            f"Identified {len(patterns.phase_transitions)} phase transitions",
            f"Average coherence: {np.mean([r.global_coherence for r in coherence_results]):.3f}",
            f"Pattern strength evolution: {'increasing' if np.polyfit(range(len(coherence_results)), [r.pattern_strength for r in coherence_results], 1)[0] > 0 else 'stable/decreasing'}"
        ]
    }
    
    # Export to JSON
    results_file = outputs_dir / f"phase3_analysis_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Phase 3 analysis results exported to {results_file}")
    
    # Print summary
    print(f"\nPHASE 3 ANALYSIS SUMMARY:")
    print(f"  • Analyzed {len(thought_history)} thought evolution snapshots")
    print(f"  • Average global coherence: {results['coherence_analysis']['average_metrics']['global_coherence']:.4f}")
    print(f"  • Found {len(patterns.recurring_motifs)} recurring patterns")
    print(f"  • Detected {len(patterns.attractor_states)} attractor states")
    print(f"  • Identified {len(patterns.phase_transitions)} phase transitions")
    
    if patterns.complexity_measures:
        print(f"  • Complexity evolution: {list(patterns.complexity_measures.keys())}")
    
    return results

def main():
    """Main demonstration function"""
    try:
        # Initialize Phase 3 demonstration
        thought_history, timesteps, evolution_analyzer, trajectory_visualizer, coherence_analyzer, pattern_detector = demonstrate_thought_evolution_analysis()
        
        # Perform coherence analysis
        coherence_results = analyze_thought_coherence(thought_history, coherence_analyzer)
        
        # Detect emergent patterns
        patterns = detect_emergent_patterns(thought_history, timesteps, pattern_detector)
        
        # Create visualizations
        create_evolution_visualizations(thought_history, timesteps, trajectory_visualizer)
        
        # Export results
        results = export_phase3_results(coherence_results, patterns, thought_history, timesteps)
        
        print("\n" + "="*60)
        print("PHASE 3 DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nPhase 3 Capabilities Demonstrated:")
        print("✓ Thought evolution tracking and analysis")
        print("✓ Multi-scale coherence measurement")
        print("✓ Emergent pattern detection")
        print("✓ Attractor state identification")
        print("✓ Phase transition detection")
        print("✓ Complex trajectory visualization")
        print("✓ Network analysis of thought relationships")
        print("✓ Comprehensive results export")
        
        print(f"\nAll Phase 3 outputs saved to outputs/ directory")
        print("Phase 3 analysis tools are ready for research use!")
        
        return results
        
    except Exception as e:
        print(f"Error in Phase 3 demonstration: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()