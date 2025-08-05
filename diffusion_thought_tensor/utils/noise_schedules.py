"""
Advanced noise scheduling for Phase 2 diffusion training
Implements multiple schedule types optimized for large models
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Union, Tuple, Optional


class NoiseScheduler:
    """Advanced noise scheduler with multiple schedule types"""
    
    def __init__(
        self, 
        num_steps: int = 1000,
        schedule_type: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon",  # "epsilon", "v_prediction", "sample"
    ):
        self.num_steps = num_steps
        self.schedule_type = schedule_type
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.prediction_type = prediction_type
        
        # Generate noise schedule
        self.betas = self._get_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # Calculate useful coefficients
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        
        # Coefficients for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        )
    
    def _get_schedule(self) -> torch.Tensor:
        """Generate noise schedule based on type"""
        if self.schedule_type == "linear":
            return torch.linspace(self.beta_start, self.beta_end, self.num_steps)
        
        elif self.schedule_type == "sqrt":
            return torch.sqrt(torch.linspace(self.beta_start, self.beta_end, self.num_steps))
        
        elif self.schedule_type == "cosine":
            # Improved cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
            steps = torch.arange(self.num_steps + 1, dtype=torch.float64) / self.num_steps
            s = 0.008  # offset to prevent singularity at t=0
            alphas_cumprod = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clamp(betas, 0, 0.999)
        
        elif self.schedule_type == "exponential":
            # Exponential schedule for smoother transitions
            steps = torch.arange(self.num_steps, dtype=torch.float64)
            betas = self.beta_start * (self.beta_end / self.beta_start) ** (steps / (self.num_steps - 1))
            return betas
        
        elif self.schedule_type == "sigmoid":
            # Sigmoid schedule for gradual transitions
            steps = torch.arange(self.num_steps, dtype=torch.float64)
            x = (steps / (self.num_steps - 1) - 0.5) * 12  # Scale to [-6, 6]
            sigmoid = torch.sigmoid(x)
            betas = self.beta_start + (self.beta_end - self.beta_start) * sigmoid
            return betas
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def add_noise(
        self, 
        original_samples: torch.Tensor, 
        noise: torch.Tensor, 
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to samples at given timesteps"""
        # Move schedules to same device as samples
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(original_samples.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(original_samples.device)
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples
    
    def get_velocity(
        self, 
        sample: torch.Tensor, 
        noise: torch.Tensor, 
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Calculate velocity parameterization v = alpha_t * noise - sigma_t * sample"""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].to(sample.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].to(sample.device)
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity
    
    def step(
        self, 
        model_output: torch.Tensor, 
        timestep: int, 
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """Perform one denoising step"""
        t = timestep
        
        # Get schedule values
        alpha_prod_t = self.alphas_cumprod[t].to(sample.device)
        alpha_prod_t_prev = self.alphas_cumprod_prev[t].to(sample.device) if t > 0 else torch.ones_like(alpha_prod_t)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        
        if self.prediction_type == "epsilon":
            # Standard epsilon prediction
            pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        elif self.prediction_type == "v_prediction":
            # v-parameterization from "Progressive Distillation for Fast Sampling"
            pred_original_sample = alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
        elif self.prediction_type == "sample":
            # Direct sample prediction
            pred_original_sample = model_output
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")
        
        # Compute coefficients for pred_original_sample x_0 and current sample x_t
        pred_original_sample_coeff = (alpha_prod_t_prev.sqrt() * self.betas[t]) / beta_prod_t
        current_sample_coeff = self.alphas[t].sqrt() * beta_prod_t_prev / beta_prod_t
        
        # Compute predicted previous sample μ_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        # Add noise for non-final steps
        if t > 0:
            variance = self.posterior_variance[t]
            if generator is not None:
                noise = torch.randn(sample.shape, generator=generator, device=sample.device)
            else:
                noise = torch.randn_like(sample)
            pred_prev_sample = pred_prev_sample + variance.sqrt() * noise
        
        return pred_prev_sample
    
    def scale_model_input(self, sample: torch.Tensor, timestep: Union[int, torch.Tensor]) -> torch.Tensor:
        """Scale model input according to schedule (for some schedulers)"""
        # For basic DDPM, no scaling needed
        return sample
    
    def get_timesteps_tensor(self, num_inference_steps: int, device: torch.device) -> torch.Tensor:
        """Get timesteps for inference with proper spacing"""
        step_ratio = self.num_steps // num_inference_steps
        timesteps = (torch.arange(0, num_inference_steps) * step_ratio).round().long()
        timesteps = torch.flip(timesteps, dims=[0])  # Reverse for denoising
        return timesteps.to(device)


class AdaptiveNoiseScheduler(NoiseScheduler):
    """Adaptive noise scheduler that adjusts based on model performance"""
    
    def __init__(self, *args, adaptation_rate: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
        self.adaptive_betas = self.betas.clone()
    
    def update_schedule(self, loss_history: list, timestep_losses: dict):
        """Update schedule based on per-timestep losses"""
        if len(loss_history) < 10:  # Need some history
            return
        
        # Calculate which timesteps are performing poorly
        avg_loss = np.mean(loss_history[-10:])
        
        for t, loss in timestep_losses.items():
            if loss > avg_loss * 1.2:  # Performing 20% worse than average
                # Slightly reduce noise at this timestep
                self.adaptive_betas[t] *= (1 - self.adaptation_rate)
            elif loss < avg_loss * 0.8:  # Performing 20% better than average
                # Slightly increase noise at this timestep
                self.adaptive_betas[t] *= (1 + self.adaptation_rate * 0.5)
        
        # Clamp to reasonable bounds
        self.adaptive_betas = torch.clamp(self.adaptive_betas, self.beta_start, self.beta_end)
        
        # Recalculate derived quantities
        self.alphas = 1.0 - self.adaptive_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # ... recalculate other derived quantities


def create_noise_scheduler(config: dict) -> NoiseScheduler:
    """Factory function to create noise scheduler from config"""
    scheduler_type = config.get('type', 'NoiseScheduler')
    
    if scheduler_type == 'AdaptiveNoiseScheduler':
        return AdaptiveNoiseScheduler(
            num_steps=config.get('num_steps', 1000),
            schedule_type=config.get('schedule_type', 'cosine'),
            beta_start=config.get('beta_start', 0.0001),
            beta_end=config.get('beta_end', 0.02),
            prediction_type=config.get('prediction_type', 'epsilon'),
            adaptation_rate=config.get('adaptation_rate', 0.01)
        )
    else:
        return NoiseScheduler(
            num_steps=config.get('num_steps', 1000),
            schedule_type=config.get('schedule_type', 'cosine'),
            beta_start=config.get('beta_start', 0.0001),
            beta_end=config.get('beta_end', 0.02),
            prediction_type=config.get('prediction_type', 'epsilon')
        )


if __name__ == "__main__":
    # Test different noise schedules
    import matplotlib.pyplot as plt
    
    schedules = ["linear", "sqrt", "cosine", "exponential", "sigmoid"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, schedule_type in enumerate(schedules):
        scheduler = NoiseScheduler(num_steps=1000, schedule_type=schedule_type)
        
        axes[i].plot(scheduler.betas.numpy(), label=f'{schedule_type}')
        axes[i].set_title(f'{schedule_type.capitalize()} Schedule')
        axes[i].set_xlabel('Timestep')
        axes[i].set_ylabel('Beta')
        axes[i].grid(True)
    
    # Plot comparison
    axes[5].clear()
    for schedule_type in schedules:
        scheduler = NoiseScheduler(num_steps=1000, schedule_type=schedule_type)
        axes[5].plot(scheduler.betas.numpy(), label=f'{schedule_type}')
    
    axes[5].set_title('All Schedules Comparison')
    axes[5].set_xlabel('Timestep')
    axes[5].set_ylabel('Beta')
    axes[5].legend()
    axes[5].grid(True)
    
    plt.tight_layout()
    plt.savefig('outputs/visualizations/noise_schedules_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Noise schedules visualization saved!")