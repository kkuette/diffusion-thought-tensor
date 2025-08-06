# Diffusion Thought Tensor

A novel approach to language modeling that integrates **diffusion models** with **thought tensor architecture**, replacing traditional autoregressive generation with iterative refinement and persistent cognitive states.

## 🧠 Core Innovation

This project explores the fusion of diffusion processes with 3D thought tensors, enabling:

- **Bidirectional Context**: Thoughts can attend to future and past simultaneously
- **Iterative Refinement**: Multiple denoising steps allow thought maturation  
- **Parallel Processing**: All positions evolve together for holistic reasoning
- **Controllable Generation**: Gradient-based steering of thought evolution
- **Persistent Memory**: Thoughts evolve across conversation turns

## 🚀 Quick Start

### Prerequisites
- **GPU**: RTX 3090 (24GB) or equivalent
- **RAM**: 32GB+ recommended
- **Python**: 3.8+

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd experiment

# Install dependencies
pip install torch>=1.13.0 transformers>=4.20.0 diffusers>=0.10.0
pip install einops tqdm pyyaml numpy wandb
pip install tensorboard matplotlib seaborn

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Basic Usage
```bash
# Train the model (starts with TinyStories dataset)
python diffusion_thought_tensor/train_dream_hf.py \
    --config diffusion_thought_tensor/configs/dream_config.yaml \
    --dataset roneneldan/TinyStories \
    --epochs 5

# Compare with baseline
python train_comparison_experiment.py \
    --baseline_config complex_baseline_config.yaml \
    --enhanced_config diffusion_thought_tensor/configs/dream_config.yaml
```

## 🏗️ Architecture

### Diffusion-Based Thought Evolution

```
Input: [Text Tokens] + [Thought Tensor Stack]
    ↓
Embedding Layer (Discrete → Continuous) 
    ↓
Diffusion Process (T timesteps)
    ↓
Denoising Network + Thought Evolution
    ↓
Output: [Refined Text] + [New Thought]
    ↓
New Thought → Push to Stack (Stack Evolution)
    ↓
Updated: [Next Input] + [Evolved Thought Stack]
```

### 3D Thought Representation

- **Spatial Dimensions**: Each thought is 2D (H×W) representing structured knowledge (It could be any dimension while it is nD-1 the dimension input)
- **Temporal Stack**: Depth dimension stores thought history over time
- **Cross-Attention**: Thoughts influence text generation through attention mechanisms
- **Compression**: 3D→2D evolution reduces dimensionality while preserving information

## ✨ Key Features

### Enhanced Thought System
- **Temporal Attention**: Multi-scale processing of thought history
- **Self-Supervised Learning**: Thoughts learn through prediction and reconstruction
- **Importance-Based Retention**: Smart memory management for thought stacks
- **Temporal Dynamics**: Velocity and acceleration tracking of thought evolution

### Advanced Masking & Generation
- **Bidirectional Attention**: Leverages [DREAM](https://github.com/DreamLM/Dream) techniques for non-causal attention
- **Progressive Unmasking**: Confidence-based token revelation during generation
- **Block-wise Processing**: Efficient masking strategies from [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM)

## 🔧 Models

### StackedDiffusionModel3D
The main model with full 3D thought tensor integration:
- ~1B parameters optimized for single GPU training
- 3D thought stack with temporal evolution
- Cross-attention between thoughts and text
- DREAM-style bidirectional attention

### BaselineDreamModel  
Control model without thought system:
- Identical architecture minus thought components
- Pure DREAM-style masked language modeling
- Used for controlled comparison experiments

## 🚀 Training

### Quick Start
```bash
# Train the main 3D thought model
python diffusion_thought_tensor/train_dream_hf.py \
    --config diffusion_thought_tensor/configs/dream_config.yaml \
    --dataset roneneldan/TinyStories \
    --epochs 5 \
    --mask_ratio 0.7

# Train baseline for comparison  
python train_comparison_experiment.py \
    --baseline_config complex_baseline_config.yaml \
    --enhanced_config diffusion_thought_tensor/configs/dream_config.yaml
```

### Training Features
- **Memory Optimized**: Gradient checkpointing and mixed precision for 24GB GPUs
- **Progressive Masking**: Dynamic mask ratio scheduling during training  
- **Thought Persistence**: Maintains thought stacks across batches
- **Multi-GPU Support**: Distributed training capabilities

## 💬 Interactive Usage

### Chat Interface
```bash
python diffusion_thought_tensor/interactive_chat.py \
    --model_path outputs/dream_checkpoints/best_dream_model.pt \
    --config diffusion_thought_tensor/configs/dream_config.yaml
```

## ⚙️ Configuration

Key configuration options in `diffusion_thought_tensor/configs/dream_config.yaml`:

```yaml
model:
  embed_dim: 720          # Model size  
  num_layers: 16          # Transformer layers
  max_seq_length: 2048    # Context length

thought_tensor:
  input_dims: [16, 16, 64]  # 3D thought dimensions
  stack_size: 8             # Thought stack depth
  self_supervised_learning: true
  temporal_dynamics: true

masking:
  mask_ratio: 0.7           # Initial masking ratio
  confidence_threshold: 0.75 # Unmasking threshold
  progressive_unmasking: true

diffusion:
  num_steps: 500            # Denoising steps
  noise_schedule: "cosine"  # Noise scheduling
```

## 📊 Expected Performance

⚠️ **Note**: These are research targets, not validated benchmarks. Current implementation focuses on proof-of-concept.

### Performance Targets
- **Perplexity**: Competitive with GPT-2 small (117M params) 
- **Generation Speed**: 100-200 tokens/second on RTX 3090
- **Memory Usage**: <24GB VRAM for training
- **Thought Coherence**: >0.8 correlation between timesteps

### Theoretical Capabilities  
- **Thought Interpolation**: Smooth blending between different reasoning states
- **Controlled Evolution**: Gradient-guided thought steering
- **Error Recovery**: Self-correction during generation
- **Bidirectional Reasoning**: Fill-in-the-blank style completion


## 🎯 Research Goals

1. **Validate Thought Contribution**: Compare thought-enhanced vs baseline models
2. **Understand Thought Evolution**: Analyze how thoughts change during diffusion
3. **Explore Novel Capabilities**: Test bidirectional reasoning and thought control
4. **Scale Efficiently**: Optimize for single-GPU research setups

## 📈 Current Status

- ✅ Core 3D thought tensor architecture implemented
- ✅ DREAM-style bidirectional attention integrated  
- ✅ Enhanced thought system with temporal dynamics
- ✅ Training pipeline with memory optimization
- ✅ Interactive chat interface
- ✅ Comprehensive analysis tools
- 🔄 Large-scale training experiments (halted)
- 🔄 Thought evolution analysis and visualization

## 📊 Current Results

⚠️ **Experimental Status**: This is active research code. Current results are preliminary.

- **Training**: Successfully trains on TinyStories dataset without memory issues
- **Baseline Comparison**: Framework implemented for controlled experiments
- **Memory Efficiency**: Runs on 24GB GPU with gradient checkpointing
- **Thought Evolution**: Basic thought tensor mechanics functional

## ⚠️ Known Limitations

- **Scale**: Currently optimized for small-scale experiments (~1B parameters)
- **Validation**: Performance claims require empirical validation
- **Dependencies**: Specific version requirements for stable training
- **Hardware**: High memory requirements limit accessibility
- **Documentation**: Some advanced features lack detailed documentation

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size in config
batch_size: 4  # or lower
gradient_accumulation_steps: 4  # increase to maintain effective batch size
```

**Import Errors**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt  # if available
# Or install manually as shown in Quick Start
```

**Training Divergence**
```bash
# Lower learning rate
learning_rate: 1e-5  # default: 3e-4
```

**Dataset Loading Issues**
```bash
# Verify internet connection for HuggingFace datasets
# Or specify local dataset path in config
```

### Getting Help
- Check existing issues in the repository
- Verify hardware requirements are met
- Review configuration files for typos

## 📚 Research Documentation

This project builds on extensive research and architectural design. The following documents provide deep technical insights:

### Core Architecture Documents
- **[Complete Project Roadmap](docs/old/complete_project_roadmap.md)** - Comprehensive implementation plan with technical stack, datasets, algorithms, and detailed phases
- **[Unified Thought Tensor Architecture](docs/old/unified_thought_tensor_arch.md)** - Meta-dimensional growth mechanism with dynamic complexity scaling and emergent cognitive behaviors
- **[Original Thought Tensor Concept](docs/old/thought_tensor_project.md)** - Foundational concept documentation with budget-conscious implementation strategies

### Technical Implementation Guides  
- **[Diffusion Research Document](docs/old/diffusion_thought_tensor_research.md)** - ~500M parameter diffusion model architecture optimized for RTX 3090, with detailed training strategies
- **[Enhanced Model Improvements](docs/old/enhanced_model_improvements.md)** - Performance optimizations, memory management, advanced training techniques, and comprehensive monitoring

### Additional Research Materials
- **[Compass Artifact](docs/old/compass_artifact_wf-9c6a1bc8-95e1-4258-a3ab-2bcbed9495a8_text_markdown.md)** - Extended research analysis and architectural considerations

These documents contain the theoretical foundations, detailed algorithms, evaluation metrics, and research methodologies underlying this implementation. They provide essential context for understanding the novel approach to persistent AI cognition through diffusion-based thought tensor evolution.

## 📚 Technical Details

### Full Requirements

**Hardware**
- **GPU**: RTX 3090 (24GB) or equivalent
- **RAM**: 32GB+ recommended
- **Storage**: ~50GB for models and data

**Dependencies**
```bash
pip install torch>=1.13.0 transformers>=4.20.0 diffusers>=0.10.0
pip install einops tqdm pyyaml numpy wandb
pip install tensorboard matplotlib seaborn
```

### External Methods
This implementation builds upon:
- **DREAM**: Bidirectional attention and masking strategies - [Repository](https://github.com/DreamLM/Dream)
- **Fast-dLLM**: Efficient generation techniques - [Repository](https://github.com/NVlabs/Fast-dLLM)

---

This research combines diffusion models' parallel processing capabilities with persistent thought states, potentially enabling new forms of AI reasoning that go beyond traditional autoregressive generation.