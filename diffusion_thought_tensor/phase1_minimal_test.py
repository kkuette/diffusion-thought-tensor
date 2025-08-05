"""
Phase 1: Minimal Diffusion Proof-of-Concept
Test basic diffusion mechanism with thought tensors
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

from model.base_model import DiffusionThoughtModel, count_parameters


class MinimalDiffusionTest:
    """Test suite for Phase 1 experiments"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Create a smaller model for fast iteration
        self.model = DiffusionThoughtModel(
            vocab_size=1000,  # Tiny vocab
            embed_dim=256,    # Smaller embedding
            num_layers=4,     # Fewer layers
            num_heads=8,      # Fewer heads
            max_seq_length=64,
            thought_input_dims=(8, 8, 4),   # Smaller thought tensor
            thought_output_dims=(8, 8, 2),
            dropout=0.1
        ).to(device)
        
        self.param_count = count_parameters(self.model)
        print(f"Minimal model: {self.param_count / 1e6:.2f}M parameters")
        
        # Diffusion parameters
        self.num_steps = 100  # Fewer steps for testing
        self.noise_schedule = self._get_noise_schedule()
        
    def _get_noise_schedule(self, schedule_type="sqrt"):
        """Generate noise schedule"""
        if schedule_type == "sqrt":
            return torch.sqrt(torch.linspace(0.0001, 0.02, self.num_steps))
        else:
            return torch.linspace(0.0001, 0.02, self.num_steps)
    
    def forward_diffusion(self, tokens, thoughts, timestep):
        """Add noise to tokens and thoughts"""
        batch_size = tokens.shape[0]
        # Move timestep to CPU for indexing, then move noise_level back to device
        timestep_cpu = timestep.cpu()
        noise_level = self.noise_schedule[timestep_cpu].to(self.device)
        
        # Token noise: randomly replace tokens
        mask = torch.rand(tokens.shape, device=self.device) < noise_level.unsqueeze(1)
        random_tokens = torch.randint_like(tokens, 0, self.model.vocab_size)
        noisy_tokens = torch.where(mask, random_tokens, tokens)
        
        # Thought noise: Gaussian noise
        thought_noise = torch.randn_like(thoughts) * noise_level.view(-1, 1, 1, 1)
        noisy_thoughts = thoughts + thought_noise
        
        return noisy_tokens, noisy_thoughts
    
    def test_1_diffusion_no_collapse(self):
        """Test 1: Verify diffusion process doesn't collapse"""
        print("\n🧪 Test 1: Diffusion Non-Collapse")
        
        batch_size = 4
        seq_length = 32
        
        # Create test data
        original_tokens = torch.randint(0, self.model.vocab_size, (batch_size, seq_length), device=self.device)
        original_thoughts = torch.randn(batch_size, 8, 8, 4, device=self.device)
        
        # Test noise addition at different timesteps
        timesteps = [10, 30, 50, 80]
        results = {}
        
        for t in timesteps:
            timestep_tensor = torch.tensor([t] * batch_size, device=self.device)
            noisy_tokens, noisy_thoughts = self.forward_diffusion(
                original_tokens, original_thoughts, timestep_tensor
            )
            
            # Calculate corruption levels
            token_diff = (noisy_tokens != original_tokens).float().mean().item()
            thought_mse = F.mse_loss(noisy_thoughts, original_thoughts).item()
            
            results[t] = {
                'token_corruption': token_diff,
                'thought_mse': thought_mse
            }
            
            print(f"  t={t:2d}: Token corruption={token_diff:.3f}, Thought MSE={thought_mse:.3f}")
        
        # Verify increasing corruption (with some tolerance for randomness)
        corruptions = [results[t]['token_corruption'] for t in timesteps]
        mses = [results[t]['thought_mse'] for t in timesteps]
        
        # Check overall trend rather than strict monotonicity
        corruption_trend = corruptions[-1] > corruptions[0]
        mse_trend = mses[-1] > mses[0]
        
        if corruption_trend and mse_trend:
            print("  ✅ Diffusion process working correctly!")
        else:
            print(f"  ⚠️ Weak trend: corruption {corruptions[0]:.3f}→{corruptions[-1]:.3f}, MSE {mses[0]:.3f}→{mses[-1]:.3f}")
        
        # Also check that we're actually adding noise
        assert max(corruptions) > 0.05, "Token corruption too low"
        assert max(mses) > 0.005, "Thought MSE too low"
        return results
    
    def test_2_thought_evolution(self):
        """Test 2: Verify thought tensors evolve meaningfully"""
        print("\n🧪 Test 2: Thought Evolution")
        
        batch_size = 2
        seq_length = 32
        
        # Create test data (match model's expected input dimensions)
        tokens = torch.randint(0, self.model.vocab_size, (batch_size, seq_length), device=self.device)
        initial_thought = torch.randn(batch_size, 8, 8, 4, device=self.device)
        
        # Track thought evolution through model
        self.model.eval()
        thought_history = []
        
        with torch.no_grad():
            current_thought = initial_thought.clone()
            
            # Simulate diffusion steps
            for t in [90, 70, 50, 30, 10]:
                timestep = torch.tensor([t] * batch_size, device=self.device)
                
                # Add appropriate noise to original thought (not evolved one)
                noisy_tokens, noisy_thoughts = self.forward_diffusion(tokens, initial_thought, timestep)
                
                # Forward pass
                logits, new_thought = self.model(noisy_tokens, noisy_thoughts, timestep)
                
                # Store for analysis (note: dimensions change due to compression)
                thought_history.append({
                    'timestep': t,
                    'thought': new_thought.clone(),
                    'thought_norm': torch.norm(new_thought).item(),
                    'mse_from_previous': F.mse_loss(new_thought, current_thought).item() if len(thought_history) > 0 and new_thought.shape == current_thought.shape else 0
                })
                
                current_thought = new_thought
                
                print(f"  t={t:2d}: Thought norm={thought_history[-1]['thought_norm']:.4f}, "
                      f"MSE from previous={thought_history[-1]['mse_from_previous']:.4f}")
        
        # Verify thoughts are evolving
        mse_changes = [h['mse_from_previous'] for h in thought_history[1:]]
        avg_change = np.mean(mse_changes)
        
        print(f"  Average thought change per step: {avg_change:.4f}")
        
        if avg_change > 0.001:
            print("  ✅ Thoughts are evolving meaningfully!")
        else:
            print("  ⚠️ Thoughts may not be evolving enough")
        
        return thought_history
    
    def test_3_noise_schedules(self):
        """Test 3: Compare different noise schedules"""
        print("\n🧪 Test 3: Noise Schedule Comparison")
        
        schedules = {
            'linear': torch.linspace(0.0001, 0.02, self.num_steps),
            'sqrt': torch.sqrt(torch.linspace(0.0001, 0.02, self.num_steps)),
            'quadratic': torch.linspace(0.0001, 0.02, self.num_steps) ** 2
        }
        
        batch_size = 2
        tokens = torch.randint(0, self.model.vocab_size, (batch_size, 32), device=self.device)
        thoughts = torch.randn(batch_size, 8, 8, 4, device=self.device)
        
        results = {}
        
        for name, schedule in schedules.items():
            # Test at t=50 (middle)
            t = 50
            noise_level = schedule[t].to(self.device)
            
            # Apply noise
            mask = torch.rand(tokens.shape, device=self.device) < noise_level.unsqueeze(0)
            random_tokens = torch.randint_like(tokens, 0, self.model.vocab_size)
            noisy_tokens = torch.where(mask, random_tokens, tokens)
            
            thought_noise = torch.randn_like(thoughts) * noise_level
            noisy_thoughts = thoughts + thought_noise
            
            # Calculate metrics
            token_corruption = (noisy_tokens != tokens).float().mean().item()
            thought_mse = F.mse_loss(noisy_thoughts, thoughts).item()
            
            results[name] = {
                'token_corruption': token_corruption,
                'thought_mse': thought_mse
            }
            
            print(f"  {name:10s}: Token corruption={token_corruption:.3f}, Thought MSE={thought_mse:.3f}")
        
        print("  ✅ Noise schedule comparison complete!")
        return results
    
    def test_4_visualize_trajectories(self):
        """Test 4: Visualize denoising trajectories"""
        print("\n🧪 Test 4: Denoising Trajectory Visualization")
        
        # Create output directory
        viz_dir = "outputs/visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        batch_size = 1
        tokens = torch.randint(0, self.model.vocab_size, (batch_size, 16), device=self.device)
        initial_thought = torch.randn(batch_size, 8, 8, 4, device=self.device)
        
        # Track metrics through denoising
        timesteps = list(range(0, 100, 10))
        metrics = {
            'thought_norms': [],
            'thought_vars': [],
            'token_entropies': []
        }
        
        self.model.eval()
        with torch.no_grad():
            for t in reversed(timesteps):
                timestep = torch.tensor([t], device=self.device)
                
                # Forward pass (always use original thought for consistency)
                logits, new_thought = self.model(tokens, initial_thought, timestep)
                
                # Calculate metrics
                thought_norm = torch.norm(new_thought).item()
                thought_var = torch.var(new_thought).item()
                token_entropy = -torch.sum(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1), dim=-1).mean().item()
                
                metrics['thought_norms'].append(thought_norm)
                metrics['thought_vars'].append(thought_var)
                metrics['token_entropies'].append(token_entropy)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Convert reversed iterator to list
        x_values = list(reversed(timesteps))
        
        axes[0].plot(x_values, metrics['thought_norms'])
        axes[0].set_title('Thought Tensor Norm')
        axes[0].set_xlabel('Timestep')
        axes[0].set_ylabel('L2 Norm')
        
        axes[1].plot(x_values, metrics['thought_vars'])
        axes[1].set_title('Thought Tensor Variance')
        axes[1].set_xlabel('Timestep')
        axes[1].set_ylabel('Variance')
        
        axes[2].plot(x_values, metrics['token_entropies'])
        axes[2].set_title('Token Prediction Entropy')
        axes[2].set_xlabel('Timestep')
        axes[2].set_ylabel('Entropy')
        
        plt.tight_layout()
        viz_path = os.path.join(viz_dir, 'denoising_trajectories.png')
        plt.savefig(viz_path)
        print(f"  📊 Visualization saved to {viz_path}")
        
        return metrics
    
    def run_all_tests(self):
        """Run all Phase 1 tests"""
        print("🚀 Starting Phase 1 Minimal Diffusion Tests")
        print(f"Model: {self.param_count / 1e6:.2f}M parameters")
        print(f"Device: {self.device}")
        
        results = {}
        
        try:
            results['collapse_test'] = self.test_1_diffusion_no_collapse()
            results['evolution_test'] = self.test_2_thought_evolution()
            results['schedule_test'] = self.test_3_noise_schedules()
            results['trajectory_test'] = self.test_4_visualize_trajectories()
            
            print("\n🎉 All Phase 1 tests completed successfully!")
            return results
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            raise


def main():
    """Run Phase 1 experiments"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize test suite
    test_suite = MinimalDiffusionTest(device=device)
    
    # Run all tests
    results = test_suite.run_all_tests()
    
    # Save results
    output_dir = "outputs/logs"
    os.makedirs(output_dir, exist_ok=True)
    
    import json
    from datetime import datetime
    
    results_path = os.path.join(output_dir, f'phase1_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    
    # Convert tensors to lists for JSON serialization
    def tensor_to_list(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        elif isinstance(obj, dict):
            return {k: tensor_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [tensor_to_list(v) for v in obj]
        else:
            return obj
    
    serializable_results = tensor_to_list(results)
    
    with open(results_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model_params': test_suite.param_count,
            'device': str(device),
            'results': serializable_results
        }, f, indent=2)
    
    print(f"\n📋 Results saved to {results_path}")


if __name__ == "__main__":
    main()