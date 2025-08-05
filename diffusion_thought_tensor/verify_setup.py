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
