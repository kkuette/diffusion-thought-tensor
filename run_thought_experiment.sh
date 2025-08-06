#!/bin/bash
# Enhanced DREAM Thought Contribution Experiment Runner
# This script runs the comprehensive experiment with DREAM-specific metrics

echo "🔬 ENHANCED DREAM THOUGHT CONTRIBUTION EXPERIMENT"
echo "================================================="
echo "This experiment will:"
echo "1. Train both thought and baseline models with identical conditions"
echo "2. Compare performance using DREAM-specific metrics (masked/unmasked accuracy)"
echo "3. Analyze thought evolution patterns and progressive unmasking effectiveness"
echo "4. Provide comprehensive statistical analysis and interpretation"
echo "5. Generate detailed visualizations and recommendations"
echo ""

# Check if conda environment exists
if ! conda env list | grep -q "diffusion-thought"; then
    echo "❌ Error: conda environment 'diffusion-thought' not found"
    echo "Please create it first with:"
    echo "conda create -n diffusion-thought python=3.10"
    echo "conda activate diffusion-thought"
    echo "pip install torch torchvision torchaudio transformers datasets pyyaml tqdm matplotlib scipy"
    exit 1
fi

# Activate environment
source activate diffusion-thought

# Set default parameters (can be overridden)
EPOCHS=${1:-5}
TRAIN_SAMPLES=${2:-1000}
EVAL_SAMPLES=${3:-200}
DATASET=${4:-"OpenWebText"}  # More complex dataset with longer dependencies

echo "📊 Experiment Parameters:"
echo "  Epochs: $EPOCHS"
echo "  Training samples: $TRAIN_SAMPLES"
echo "  Evaluation samples: $EVAL_SAMPLES"
echo "  Dataset: $DATASET"
echo ""

# Test baseline model first
echo "🧪 Testing baseline model..."
python diffusion_thought_tensor/model/baseline_dream_model.py
if [ $? -ne 0 ]; then
    echo "❌ Baseline model test failed"
    exit 1
fi
echo "✅ Baseline model test passed"
echo ""

# Run the comparison experiment
echo "🏃 Starting comparison training..."
python train_comparison_experiment.py \
    --thought_config complex_dataset_config.yaml \
    --baseline_config complex_baseline_config.yaml \
    --epochs $EPOCHS \
    --train_samples $TRAIN_SAMPLES \
    --eval_samples $EVAL_SAMPLES \
    --dataset "$DATASET" \
    --seed 42

if [ $? -ne 0 ]; then
    echo "❌ Training experiment failed"
    exit 1
fi

# Find the latest experiment directory
LATEST_EXPERIMENT=$(ls -td outputs/comparison_experiment_* 2>/dev/null | head -n1)

if [ -z "$LATEST_EXPERIMENT" ]; then
    echo "❌ No experiment results found"
    exit 1
fi

echo ""
echo "📊 Running enhanced analysis on: $LATEST_EXPERIMENT"

# Create analysis output directory
ANALYSIS_DIR="$LATEST_EXPERIMENT/analysis"
mkdir -p "$ANALYSIS_DIR"

# Run enhanced results analysis
echo "🔍 Generating comprehensive analysis report..."
python analyze_enhanced_results.py \
    "$LATEST_EXPERIMENT" \
    --output "$ANALYSIS_DIR"

if [ $? -ne 0 ]; then
    echo "❌ Enhanced analysis failed"
    exit 1
fi

# Run statistical analysis
echo "📈 Running statistical significance tests..."
python statistical_analysis.py \
    "$LATEST_EXPERIMENT" \
    --output "$ANALYSIS_DIR/statistical_analysis.json"

if [ $? -ne 0 ]; then
    echo "❌ Statistical analysis failed"
    exit 1
fi

echo ""
echo "✅ ENHANCED EXPERIMENT COMPLETE!"
echo "📁 Results saved to: $LATEST_EXPERIMENT"
echo ""
echo "📋 Key files to check:"
echo "  📊 Core Results:"
echo "    - $LATEST_EXPERIMENT/final_results.json (comprehensive results with thought metrics)"
echo "    - $LATEST_EXPERIMENT/experiment_info.json (experiment configuration)"
echo ""
echo "  📈 Analysis & Statistics:"
echo "    - $ANALYSIS_DIR/comprehensive_analysis.json (detailed DREAM analysis)"
echo "    - $ANALYSIS_DIR/statistical_analysis.json (significance tests)"
echo ""
echo "  📊 Visualizations:"
echo "    - $ANALYSIS_DIR/performance_comparison.png (overall performance trends)"
echo "    - $ANALYSIS_DIR/dream_metrics.png (DREAM-specific metrics)"
echo "    - $ANALYSIS_DIR/thought_evolution.png (thought evolution patterns)"
echo ""
echo "🎯 INTERPRETATION:"
echo "Check comprehensive_analysis.json for:"
echo "  • Overall assessment of thought contribution"
echo "  • DREAM-specific benefits (masked vs unmasked accuracy)"
echo "  • Thought evolution patterns and effectiveness"
echo "  • Statistical significance of improvements"
echo "  • Actionable recommendations for next steps"
echo ""
echo "🔍 For quick results, look for 'overall_assessment' and 'interpretation' fields!"