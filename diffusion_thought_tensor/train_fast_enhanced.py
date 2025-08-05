"""
Training script for Fast-dLLM Enhanced Diffusion Thought Tensor Model
Demonstrates performance improvements with KV caching and parallel decoding
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import time
import json
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from model.fast_enhanced_model import FastDiffusionThoughtModel, count_parameters
    from model.enhanced_model import EnhancedDiffusionThoughtModel
    from utils.noise_schedules import create_noise_scheduler
except ImportError:
    import sys
    sys.path.append('.')
    from model.fast_enhanced_model import FastDiffusionThoughtModel, count_parameters
    from model.enhanced_model import EnhancedDiffusionThoughtModel
    from utils.noise_schedules import create_noise_scheduler


class SyntheticTextDataset(Dataset):
    """Synthetic dataset for testing Fast-dLLM optimizations"""
    
    def __init__(self, vocab_size=1000, seq_length=128, num_samples=10000):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
        
        # Generate synthetic sequences with some structure
        self.data = []
        for _ in range(num_samples):
            # Create sequences with repeating patterns for better caching
            pattern_length = np.random.randint(4, 16)
            pattern = torch.randint(0, vocab_size, (pattern_length,))
            
            sequence = []
            while len(sequence) < seq_length:
                remaining = seq_length - len(sequence)
                if remaining >= pattern_length:
                    sequence.extend(pattern.tolist())
                else:
                    sequence.extend(pattern[:remaining].tolist())
            
            self.data.append(torch.tensor(sequence))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


def benchmark_model_performance(model, dataloader, device, num_batches=50, model_name="Model"):
    """Benchmark model inference performance"""
    model.eval()
    model.to(device)
    
    total_time = 0
    total_tokens = 0
    batch_times = []
    
    print(f"\n🚀 Benchmarking {model_name} Performance...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc=f"Benchmarking {model_name}")):
            if i >= num_batches:
                break
            
            batch = batch.to(device)
            batch_size, seq_length = batch.shape
            
            # Initialize thought stack
            thought_stack = model.initialize_thought_stack(batch_size, device)
            timestep = torch.randint(0, 500, (batch_size,)).to(device)
            
            # Time the forward pass
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            
            if hasattr(model, 'fast_generate'):
                # Use fast generation for Fast-dLLM model
                tokens, final_stack, stats = model.fast_generate(
                    initial_stack=thought_stack,
                    prompt_tokens=batch[:, :seq_length//2],
                    num_steps=25,
                    use_parallel_decoding=True
                )
            else:
                # Standard forward pass for baseline model
                logits, new_thought = model(batch, thought_stack, timestep)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            batch_time = end_time - start_time
            total_time += batch_time
            total_tokens += batch_size * seq_length
            batch_times.append(batch_time)
    
    avg_time_per_batch = total_time / num_batches
    tokens_per_second = total_tokens / total_time
    throughput = num_batches * dataloader.batch_size / total_time
    
    results = {
        'model_name': model_name,
        'avg_time_per_batch': avg_time_per_batch,
        'tokens_per_second': tokens_per_second,
        'throughput_samples_per_sec': throughput,
        'total_time': total_time,
        'batch_times': batch_times
    }
    
    print(f"📊 {model_name} Results:")
    print(f"  • Average time per batch: {avg_time_per_batch:.4f}s")
    print(f"  • Tokens per second: {tokens_per_second:.1f}")
    print(f"  • Throughput: {throughput:.1f} samples/sec")
    
    return results


def compare_models():
    """Compare Fast-dLLM enhanced model with baseline"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Model configurations
    config = {
        'vocab_size': 1000,
        'embed_dim': 256,
        'num_layers': 6,
        'num_heads': 8,
        'max_seq_length': 128,
        'thought_dims': (16, 16, 8),
        'thought_stack_size': 8,
        'dropout': 0.1
    }
    
    # Create models
    print("🏗️  Creating models...")
    baseline_model = EnhancedDiffusionThoughtModel(**config)
    fast_model = FastDiffusionThoughtModel(
        **config,
        use_kv_cache=True,
        confidence_threshold=0.8,
        block_size=32
    )
    
    baseline_params = count_parameters(baseline_model)
    fast_params = count_parameters(fast_model)
    
    print(f"📏 Model sizes:")
    print(f"  • Baseline: {baseline_params / 1e6:.2f}M parameters")
    print(f"  • Fast-dLLM: {fast_params / 1e6:.2f}M parameters")
    
    # Create dataset
    dataset = SyntheticTextDataset(vocab_size=1000, seq_length=128, num_samples=1000)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Benchmark both models
    baseline_results = benchmark_model_performance(
        baseline_model, dataloader, device, num_batches=30, model_name="Baseline"
    )
    
    fast_results = benchmark_model_performance(
        fast_model, dataloader, device, num_batches=30, model_name="Fast-dLLM"
    )
    
    # Calculate speedup
    speedup_throughput = fast_results['throughput_samples_per_sec'] / baseline_results['throughput_samples_per_sec']
    speedup_tokens = fast_results['tokens_per_second'] / baseline_results['tokens_per_second']
    
    print(f"\n🎯 Performance Improvements:")
    print(f"  • Throughput speedup: {speedup_throughput:.2f}x")
    print(f"  • Token processing speedup: {speedup_tokens:.2f}x")
    print(f"  • Time reduction: {(1 - fast_results['avg_time_per_batch'] / baseline_results['avg_time_per_batch']) * 100:.1f}%")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'config': config,
        'baseline_results': baseline_results,
        'fast_results': fast_results,
        'improvements': {
            'throughput_speedup': speedup_throughput,
            'token_speedup': speedup_tokens,
            'time_reduction_percent': (1 - fast_results['avg_time_per_batch'] / baseline_results['avg_time_per_batch']) * 100
        }
    }
    
    # Save results
    os.makedirs('outputs/fast_benchmarks', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'outputs/fast_benchmarks/fast_dllm_comparison_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_copy = json.loads(json.dumps(results, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x))
        json.dump(results_copy, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")
    
    return results


def test_caching_effectiveness():
    """Test KV caching effectiveness"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with caching
    model = FastDiffusionThoughtModel(
        vocab_size=1000,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        max_seq_length=128,
        use_kv_cache=True
    ).to(device)
    
    print("🧪 Testing KV Cache Effectiveness...")
    
    # Test cache utilization
    batch_size = 4
    seq_length = 64
    
    # Generate test data
    tokens = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
    thought_stack = model.initialize_thought_stack(batch_size, device)
    timestep = torch.randint(0, 500, (batch_size,)).to(device)
    
    model.eval()
    with torch.no_grad():
        # First pass - builds cache
        start_time = time.time()
        logits1, _, _ = model(tokens, thought_stack, timestep, use_cache=True)
        first_pass_time = time.time() - start_time
        
        # Second pass - should use cache (simulate incremental decoding)
        start_time = time.time()
        logits2, _, _ = model(tokens, thought_stack, timestep, use_cache=True)
        second_pass_time = time.time() - start_time
        
        # Third pass without cache
        model.clear_caches()
        start_time = time.time()
        logits3, _, _ = model(tokens, thought_stack, timestep, use_cache=False)
        no_cache_time = time.time() - start_time
    
    print(f"⏱️  Cache Performance:")
    print(f"  • First pass (build cache): {first_pass_time:.4f}s")
    print(f"  • Second pass (use cache): {second_pass_time:.4f}s")
    print(f"  • No cache pass: {no_cache_time:.4f}s")
    print(f"  • Cache speedup: {no_cache_time / second_pass_time:.2f}x")
    
    # Verify cache is working by checking cache sizes
    cache_info = []
    for i, block in enumerate(model.transformer_blocks):
        if block.kv_cache:
            cache_size = block.kv_cache.get_cache_size()
            cache_info.append(f"Layer {i}: {cache_size} tokens cached")
    
    if cache_info:
        print(f"📦 Cache Status:")
        for info in cache_info[:3]:  # Show first 3 layers
            print(f"    {info}")
    
    return {
        'first_pass_time': first_pass_time,
        'cached_pass_time': second_pass_time,
        'no_cache_time': no_cache_time,
        'cache_speedup': no_cache_time / second_pass_time
    }


def demonstrate_parallel_decoding():
    """Demonstrate confidence-aware parallel decoding"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = FastDiffusionThoughtModel(
        vocab_size=1000,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        confidence_threshold=0.7,
        use_kv_cache=True
    ).to(device)
    
    print("🔀 Testing Parallel Decoding...")
    
    # Create test prompt
    prompt = torch.randint(0, 1000, (1, 20)).to(device)
    
    model.eval()
    with torch.no_grad():
        # Sequential decoding (baseline)
        start_time = time.time()
        tokens_seq, _, _ = model.fast_generate(
            prompt_tokens=prompt,
            num_steps=50,
            use_parallel_decoding=False
        )
        sequential_time = time.time() - start_time
        
        # Parallel decoding
        start_time = time.time()
        tokens_par, _, stats = model.fast_generate(
            prompt_tokens=prompt,
            num_steps=50,
            use_parallel_decoding=True
        )
        parallel_time = time.time() - start_time
    
    print(f"⚡ Parallel Decoding Results:")
    print(f"  • Sequential time: {sequential_time:.4f}s")
    print(f"  • Parallel time: {parallel_time:.4f}s")
    print(f"  • Speedup: {sequential_time / parallel_time:.2f}x")
    print(f"  • Parallel acceptance rate: {stats['parallel_acceptance_rate']:.2f}")
    print(f"  • Total parallel accepts: {stats['total_parallel_accepts']}")
    
    return {
        'sequential_time': sequential_time,
        'parallel_time': parallel_time,
        'speedup': sequential_time / parallel_time,
        'acceptance_rate': stats['parallel_acceptance_rate']
    }


if __name__ == "__main__":
    print("🚀 Fast-dLLM Enhanced Diffusion Thought Tensor Model Benchmark")
    print("=" * 70)
    
    try:
        # Test individual components
        print("\n1️⃣  Testing KV Cache Effectiveness")
        cache_results = test_caching_effectiveness()
        
        print("\n2️⃣  Testing Parallel Decoding")
        parallel_results = demonstrate_parallel_decoding()
        
        print("\n3️⃣  Comprehensive Model Comparison")
        comparison_results = compare_models()
        
        print("\n✅ All benchmarks completed successfully!")
        print("\n🎉 Summary of Improvements:")
        print(f"  • KV Cache speedup: {cache_results['cache_speedup']:.2f}x")
        print(f"  • Parallel decoding speedup: {parallel_results['speedup']:.2f}x")
        print(f"  • Overall throughput improvement: {comparison_results['improvements']['throughput_speedup']:.2f}x")
        
    except Exception as e:
        print(f"❌ Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()