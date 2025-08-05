"""
Visualize Thought Stack Evolution
Shows how thought tensors evolve during text generation
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import yaml
import os
import argparse
from typing import List, Tuple, Dict
from transformers import AutoTokenizer

# Import our models
from model.enhanced_model import EnhancedDiffusionThoughtModel
from utils.noise_schedules import create_noise_scheduler


class ThoughtVisualizer:
    """Visualizes thought stack evolution during generation"""
    
    def __init__(self, model_path: str, config_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize model
        self.model = EnhancedDiffusionThoughtModel(
            vocab_size=self.config['model'].get('vocab_size', 50257),
            embed_dim=self.config['model']['embed_dim'],
            num_layers=self.config['model']['num_layers'],
            num_heads=self.config['model']['num_heads'],
            max_seq_length=self.config['model']['max_seq_length'],
            thought_dims=tuple(self.config['thought_tensor']['input_dims']),
            thought_stack_size=self.config['thought_tensor']['stack_size'],
            dropout=self.config['model']['dropout']
        )
        
        # Load trained weights
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device).eval()
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded on {self.device}")
        print(f"Thought dimensions: {self.model.thought_dims}")
        print(f"Stack size: {self.model.thought_stack_size}")
    
    def generate_with_tracking(
        self, 
        prompt: str = "Once upon a time", 
        max_length: int = 50,
        temperature: float = 0.8
    ) -> Tuple[str, List[torch.Tensor], List[str]]:
        """Generate text while tracking thought evolution"""
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        batch_size, seq_len = input_ids.shape
        
        # Initialize thought stack
        thought_stack = self.model.initialize_thought_stack(batch_size, self.device)
        
        # Track evolution
        thought_history = []
        token_history = []
        generated_text = prompt
        
        print(f"Starting generation with prompt: '{prompt}'")
        
        with torch.no_grad():
            current_tokens = input_ids
            
            for step in range(max_length):
                # Create dummy timestep (not used in language modeling mode)
                timestep = torch.zeros(batch_size, device=self.device, dtype=torch.long)
                
                # Forward pass
                logits, new_thought = self.model(current_tokens, thought_stack, timestep)
                
                # Sample next token
                if temperature > 0:
                    probs = F.softmax(logits[:, -1] / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                
                # Decode token
                token_text = self.tokenizer.decode(next_token[0].item())
                token_history.append(token_text)
                generated_text += token_text
                
                # Store current thought stack state
                thought_history.append(thought_stack.clone().cpu())
                
                # Evolve thought stack
                thought_stack = self.model.evolve_thought_stack(thought_stack, new_thought)
                
                # Prepare for next iteration
                current_tokens = torch.cat([current_tokens, next_token], dim=1)
                
                # Truncate if getting too long
                if current_tokens.shape[1] > self.model.max_seq_length:
                    current_tokens = current_tokens[:, -self.model.max_seq_length:]
                
                print(f"Step {step+1}: '{token_text}' | Thought norm: {new_thought.norm().item():.3f}")
        
        return generated_text, thought_history, token_history
    
    def visualize_thought_evolution(
        self, 
        thought_history: List[torch.Tensor], 
        token_history: List[str],
        save_path: str = "thought_evolution.png"
    ):
        """Create comprehensive visualization of thought evolution"""
        
        num_steps = len(thought_history)
        stack_size = thought_history[0].shape[1]
        thought_dims = thought_history[0].shape[2:]
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Thought stack norms over time
        ax1 = fig.add_subplot(gs[0, :2])
        stack_norms = []
        for i, stack in enumerate(thought_history):
            # Compute norm for each thought in the stack
            norms = []
            for j in range(stack.shape[0]):
                thought = stack[j]
                norm = torch.norm(thought.flatten()).item()
                norms.append(norm)
            norms = np.array(norms)
            stack_norms.append(norms)
        
        stack_norms = np.array(stack_norms)
        
        # Plot each position in stack
        for pos in range(stack_size):
            ax1.plot(stack_norms[:, pos], label=f'Stack pos {pos}', alpha=0.7)
        
        ax1.set_xlabel('Generation Step')
        ax1.set_ylabel('Thought Norm')
        ax1.set_title('Thought Stack Norms Over Time')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Latest thought evolution (2D projection)
        ax2 = fig.add_subplot(gs[0, 2:])
        latest_thoughts = [stack[0, -1].flatten().numpy() for stack in thought_history]
        latest_thoughts = np.array(latest_thoughts)
        
        # PCA for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        thoughts_2d = pca.fit_transform(latest_thoughts)
        
        scatter = ax2.scatter(thoughts_2d[:, 0], thoughts_2d[:, 1], 
                            c=range(len(thoughts_2d)), cmap='viridis', s=50)
        
        # Draw trajectory
        ax2.plot(thoughts_2d[:, 0], thoughts_2d[:, 1], 'r-', alpha=0.5, linewidth=1)
        
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
        ax2.set_title('Latest Thought Evolution (PCA)')
        plt.colorbar(scatter, ax=ax2, label='Generation Step')
        
        # 3. Thought similarity matrix
        ax3 = fig.add_subplot(gs[1, :2])
        
        # Compute cosine similarities between consecutive thoughts
        similarities = []
        for i in range(1, len(thought_history)):
            prev_thought = thought_history[i-1][0, -1].flatten()
            curr_thought = thought_history[i][0, -1].flatten()
            sim = F.cosine_similarity(prev_thought, curr_thought, dim=0).item()
            similarities.append(sim)
        
        ax3.plot(similarities, 'b-', linewidth=2)
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Generation Step')
        ax3.set_ylabel('Cosine Similarity')
        ax3.set_title('Thought Similarity (Consecutive Steps)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Token-thought correlation
        ax4 = fig.add_subplot(gs[1, 2:])
        
        # Compute change in thought vs token novelty
        thought_changes = []
        for i in range(1, len(thought_history)):
            prev_thought = thought_history[i-1][0, -1].flatten()
            curr_thought = thought_history[i][0, -1].flatten()
            change = torch.norm(curr_thought - prev_thought).item()
            thought_changes.append(change)
        
        ax4.bar(range(len(thought_changes)), thought_changes, alpha=0.7)
        ax4.set_xlabel('Generation Step')
        ax4.set_ylabel('Thought Change Magnitude')
        ax4.set_title('Thought Changes per Token')
        
        # Add token labels
        for i, token in enumerate(token_history[1:]):
            if i < len(thought_changes):
                ax4.text(i, thought_changes[i], token, 
                        rotation=45, ha='left', va='bottom', fontsize=8)
        
        # 5. Thought heatmap (average across spatial dimensions)
        ax5 = fig.add_subplot(gs[2, :])
        
        # Average thought activations over spatial dimensions
        thought_activations = []
        for stack in thought_history:
            # Take latest thought and average over spatial dims
            latest_thought = stack[0, -1]  # Shape: (32, 32, 16)
            # Average over H and W dimensions (0 and 1)
            activation = latest_thought.mean(dim=(0, 1)).numpy()  # Average over H, W -> shape (16,)
            thought_activations.append(activation)
        
        thought_activations = np.array(thought_activations).T  # Time x Channels -> Channels x Time
        
        sns.heatmap(thought_activations, ax=ax5, cmap='coolwarm', center=0,
                   xticklabels=[f'{i}:{token}' for i, token in enumerate(token_history)],
                   yticklabels=[f'Ch{i}' for i in range(thought_activations.shape[0])],
                   cbar_kws={'label': 'Activation'})
        
        ax5.set_xlabel('Generation Step : Token')
        ax5.set_ylabel('Thought Channel')
        ax5.set_title('Thought Channel Activations Over Time')
        
        plt.suptitle(f'Thought Stack Evolution During Generation', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
        
        return fig
    
    def create_animation(
        self, 
        thought_history: List[torch.Tensor],
        token_history: List[str],
        save_path: str = "thought_animation.gif"
    ):
        """Create animated visualization of thought evolution"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            if frame >= len(thought_history):
                return
            
            stack = thought_history[frame][0]  # First batch item
            
            # Left plot: Current thought stack visualization
            stack_2d = stack.mean(dim=-1)  # Average over depth dimension
            stack_reshaped = stack_2d.view(stack.shape[0], -1).numpy()
            
            im1 = ax1.imshow(stack_reshaped, cmap='coolwarm', aspect='auto')
            ax1.set_title(f'Step {frame}: Thought Stack\nToken: "{token_history[frame] if frame < len(token_history) else ""}"')
            ax1.set_xlabel('Spatial Position')
            ax1.set_ylabel('Stack Position')
            
            # Right plot: Thought norms evolution
            norms_so_far = []
            for i in range(min(frame + 1, len(thought_history))):
                stack_norms = torch.norm(thought_history[i][0], dim=(1, 2, 3)).numpy()
                norms_so_far.append(stack_norms)
            
            if norms_so_far:
                norms_array = np.array(norms_so_far)
                for pos in range(norms_array.shape[1]):
                    ax2.plot(norms_array[:, pos], label=f'Stack {pos}' if frame == 0 else "")
            
            ax2.set_title('Thought Norms Over Time')
            ax2.set_xlabel('Generation Step')
            ax2.set_ylabel('Norm')
            ax2.grid(True, alpha=0.3)
            if frame == 0:
                ax2.legend()
        
        ani = FuncAnimation(fig, animate, frames=len(thought_history), interval=500, repeat=True)
        ani.save(save_path, writer='pillow', fps=2)
        print(f"Animation saved to {save_path}")
        
        return ani


def main():
    parser = argparse.ArgumentParser(description="Visualize thought stack evolution")
    parser.add_argument("--model_path", default="outputs/enhanced_checkpoints/best_enhanced_model.pt", 
                       help="Path to trained model")
    parser.add_argument("--config", default="configs/enhanced_config.yaml", 
                       help="Config file path")
    parser.add_argument("--prompt", default="Once upon a time", 
                       help="Text prompt for generation")
    parser.add_argument("--max_length", type=int, default=30, 
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8, 
                       help="Sampling temperature")
    parser.add_argument("--output_dir", default="visualizations", 
                       help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize visualizer
    try:
        visualizer = ThoughtVisualizer(args.model_path, args.config)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Generate text with thought tracking
    print("\n" + "="*60)
    print("GENERATING TEXT WITH THOUGHT TRACKING")
    print("="*60)
    
    generated_text, thought_history, token_history = visualizer.generate_with_tracking(
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature
    )
    
    print(f"\nGenerated text:\n'{generated_text}'")
    print(f"\nGenerated {len(token_history)} tokens")
    print(f"Tracked {len(thought_history)} thought states")
    
    # Create visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Static visualization
    static_path = os.path.join(args.output_dir, "thought_evolution.png")
    visualizer.visualize_thought_evolution(thought_history, token_history, static_path)
    
    # Animated visualization
    animation_path = os.path.join(args.output_dir, "thought_animation.gif")
    try:
        visualizer.create_animation(thought_history, token_history, animation_path)
    except Exception as e:
        print(f"Could not create animation: {e}")
    
    print(f"\nVisualization complete! Check {args.output_dir}/ for outputs")


if __name__ == "__main__":
    main()