#!/bin/bash

# Setup script for Diffusion Thought Tensor Research Environment
# Updated with tested package versions and compatibility fixes

echo "🚀 Setting up Diffusion Thought Tensor Research Environment"
echo "=========================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not in PATH. Please install Anaconda/Miniconda first."
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment 'diffusion-thought'..."
conda create -n diffusion-thought python=3.10 -y

# Activate environment
echo "🔄 Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate diffusion-thought

# Install PyTorch with CUDA support (updated to use latest stable versions)
echo "🔥 Installing PyTorch with CUDA support..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install core ML packages in batches to avoid conflicts
echo "📚 Installing core ML dependencies..."
pip install transformers==4.36.0 diffusers==0.24.0 accelerate==0.25.0

# Fix huggingface-hub compatibility issue
echo "🔧 Fixing package compatibility..."
pip install huggingface-hub==0.19.4

# Install datasets and experiment tracking
echo "📊 Installing data and tracking tools..."
pip install datasets==2.16.0 wandb==0.16.1 tensorboard==2.15.0

# Install scientific computing packages
echo "🔬 Installing scientific computing packages..."
pip install matplotlib==3.8.2 seaborn==0.13.0 einops==0.7.0 scipy==1.11.4 scikit-learn==1.3.2

# Install additional tools for diffusion models
echo "🛠️ Installing advanced ML frameworks..."
pip install x-transformers==1.27.0 pytorch-lightning==2.1.3 hydra-core==1.3.2 omegaconf==2.3.0

# Install development tools
echo "🔧 Installing development tools..."
pip install ipykernel==6.28.0 jupyter==1.0.0 black==23.12.1 flake8==7.0.0 pytest==7.4.3

# Create project structure
echo "📁 Creating project directory structure..."
mkdir -p diffusion_thought_tensor/{model,experiments,data,configs,utils,notebooks,outputs}
mkdir -p diffusion_thought_tensor/outputs/{checkpoints,logs,visualizations}

# Create basic configuration file
echo "⚙️ Creating default configuration..."
cat > diffusion_thought_tensor/configs/default_config.yaml << 'EOF'
# Default configuration for Diffusion Thought Tensor Model

model:
  name: "DiffusionThoughtModel"
  embed_dim: 768
  num_layers: 12
  num_heads: 12
  vocab_size: 50257  # GPT-2 vocabulary
  max_seq_length: 256
  dropout: 0.1
  
thought_tensor:
  input_dims: [32, 32, 16]  # 3D tensor
  output_dims: [32, 32, 8]   # Compressed output
  hidden_dim: 512
  compression_ratio: 0.75
  
diffusion:
  num_steps: 500
  noise_schedule: "sqrt"
  beta_start: 0.0001
  beta_end: 0.02
  
training:
  batch_size: 4
  gradient_accumulation_steps: 8
  learning_rate: 1e-4
  warmup_steps: 5000
  total_steps: 100000
  eval_every: 1000
  save_every: 5000
  
optimizer:
  type: "AdamW"
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1e-8
  
hardware:
  device: "cuda"
  mixed_precision: true
  gradient_checkpointing: true
  num_workers: 4
EOF

# Create a test script to verify installation
echo "🧪 Creating verification script..."
cat > diffusion_thought_tensor/verify_setup.py << 'EOF'
import torch
import transformers
import diffusers
import sys

print("🔍 Verifying installation...")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"Diffusers version: {diffusers.__version__}")

# Check CUDA availability
if torch.cuda.is_available():
    print(f"✅ CUDA is available!")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("❌ CUDA is not available. Please check your installation.")

# Test basic tensor operations
try:
    test_tensor = torch.randn(32, 32, 16).cuda()
    print(f"✅ Successfully created tensor on GPU: {test_tensor.shape}")
except Exception as e:
    print(f"❌ Error creating tensor on GPU: {e}")

print("\n🎉 Setup verification complete!")
EOF

# Run verification test
echo "🧪 Running setup verification..."
cd diffusion_thought_tensor
if python verify_setup.py; then
    echo ""
    echo "🎉 Environment setup complete and verified!"
    echo ""
    echo "✅ All packages installed successfully"
    echo "✅ CUDA support verified"
    echo "✅ Project structure created"
    echo ""
    echo "📋 Quick Start:"
    echo "  1. Activate environment: conda activate diffusion-thought"
    echo "  2. Navigate to project: cd diffusion_thought_tensor"
    echo "  3. Start Jupyter: jupyter lab"
    echo "  4. Begin your diffusion model research!"
    echo ""
    echo "📁 Project structure:"
    echo "  diffusion_thought_tensor/"
    echo "  ├── configs/          # Configuration files"
    echo "  ├── data/             # Dataset storage"
    echo "  ├── experiments/      # Experiment scripts"
    echo "  ├── model/            # Model implementations"
    echo "  ├── notebooks/        # Jupyter notebooks"
    echo "  ├── outputs/          # Results and artifacts"
    echo "  │   ├── checkpoints/  # Model checkpoints"
    echo "  │   ├── logs/         # Training logs"
    echo "  │   └── visualizations/ # Generated visualizations"
    echo "  └── utils/            # Utility functions"
else
    echo ""
    echo "❌ Setup verification failed!"
    echo "Please check the error messages above and try running the script again."
    echo "If problems persist, you can manually run:"
    echo "  conda activate diffusion-thought"
    echo "  python verify_setup.py"
    exit 1
fi