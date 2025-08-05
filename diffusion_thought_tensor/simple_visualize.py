"""
Simple Thought Stack Visualization
Shows how thought tensors evolve during text generation
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import argparse
from transformers import AutoTokenizer

# Import our models
from model.enhanced_model import EnhancedDiffusionThoughtModel


def load_model(model_path, config_path):
    """Load trained model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize model
    model = EnhancedDiffusionThoughtModel(
        vocab_size=config['model'].get('vocab_size', 50257),
        embed_dim=config['model']['embed_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        max_seq_length=config['model']['max_seq_length'],
        thought_dims=tuple(config['thought_tensor']['input_dims']),
        thought_stack_size=config['thought_tensor']['stack_size'],
        dropout=0.0  # No dropout for inference
    )
    
    # Load trained weights
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device).eval()
    
    return model, device, config


def generate_and_track(model, device, prompt="The cat", max_length=20, temperature=0.8):
    """Generate text while tracking thought evolution"""
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    batch_size = 1
    
    # Initialize thought stack
    thought_stack = model.initialize_thought_stack(batch_size, device)
    
    # Track evolution
    thought_norms = []
    thought_changes = []
    tokens_generated = []
    
    print(f"\nGenerating from prompt: '{prompt}'")
    print("-" * 60)
    
    with torch.no_grad():
        current_tokens = input_ids
        prev_thought = None
        
        for step in range(max_length):
            # Forward pass
            timestep = torch.zeros(batch_size, device=device, dtype=torch.long)
            logits, new_thought = model(current_tokens, thought_stack, timestep)
            
            # Sample next token
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Decode token
            token_text = tokenizer.decode(next_token[0].item())
            tokens_generated.append(token_text)
            
            # Track thought metrics
            norm = torch.norm(new_thought).item()
            thought_norms.append(norm)
            
            if prev_thought is not None:
                change = torch.norm(new_thought - prev_thought).item()
                thought_changes.append(change)
            
            prev_thought = new_thought.clone()
            
            # Evolve thought stack
            thought_stack = model.evolve_thought_stack(thought_stack, new_thought)
            
            # Prepare for next iteration
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
            if current_tokens.shape[1] > model.max_seq_length:
                current_tokens = current_tokens[:, -model.max_seq_length:]
            
            print(f"Step {step+1:2d}: '{token_text:20s}' | Norm: {norm:6.2f} | "
                  f"Change: {thought_changes[-1] if thought_changes else 0:6.2f}")
    
    # Get final text
    final_text = tokenizer.decode(current_tokens[0], skip_special_tokens=False)
    
    return final_text, thought_norms, thought_changes, tokens_generated


def visualize_simple(thought_norms, thought_changes, tokens, save_path="thought_vis_simple.png"):
    """Create simple visualization of thought evolution"""
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. Thought norms over time
    ax1.plot(thought_norms, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Generation Step')
    ax1.set_ylabel('Thought Norm')
    ax1.set_title('Thought Tensor Norms During Generation')
    ax1.grid(True, alpha=0.3)
    
    # Add token annotations
    for i, token in enumerate(tokens[:10]):  # Show first 10 tokens
        ax1.annotate(token, (i, thought_norms[i]), 
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=8, rotation=45)
    
    # 2. Thought changes
    if thought_changes:
        ax2.bar(range(len(thought_changes)), thought_changes, alpha=0.7, color='green')
        ax2.set_xlabel('Generation Step')
        ax2.set_ylabel('Change Magnitude')
        ax2.set_title('Thought Changes Between Steps')
        ax2.grid(True, alpha=0.3)
    
    # 3. Rolling statistics
    window = 5
    if len(thought_norms) > window:
        rolling_mean = np.convolve(thought_norms, np.ones(window)/window, mode='valid')
        rolling_std = [np.std(thought_norms[i:i+window]) for i in range(len(thought_norms)-window+1)]
        
        ax3.plot(range(window-1, len(thought_norms)), rolling_mean, 'r-', label='Rolling Mean', linewidth=2)
        ax3.fill_between(range(window-1, len(thought_norms)), 
                        rolling_mean - np.array(rolling_std), 
                        rolling_mean + np.array(rolling_std), 
                        alpha=0.3, color='red', label='±1 Std Dev')
        ax3.set_xlabel('Generation Step')
        ax3.set_ylabel('Thought Norm')
        ax3.set_title(f'Rolling Statistics (window={window})')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Simple thought stack visualization")
    parser.add_argument("--model_path", default="outputs/enhanced_checkpoints/best_enhanced_model.pt")
    parser.add_argument("--config", default="configs/enhanced_config.yaml")
    parser.add_argument("--prompt", default="The cat")
    parser.add_argument("--max_length", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.8)
    
    args = parser.parse_args()
    
    # Load model
    model, device, config = load_model(args.model_path, args.config)
    print(f"Model loaded on {device}")
    print(f"Thought dimensions: {model.thought_dims}")
    print(f"Stack size: {model.thought_stack_size}")
    
    # Generate and track
    final_text, thought_norms, thought_changes, tokens = generate_and_track(
        model, device, args.prompt, args.max_length, args.temperature
    )
    
    print(f"\n{'='*60}")
    print(f"Final generated text:\n'{final_text}'")
    print(f"{'='*60}")
    
    # Analysis
    print(f"\nThought Evolution Analysis:")
    print(f"- Average thought norm: {np.mean(thought_norms):.2f}")
    print(f"- Std dev of norms: {np.std(thought_norms):.2f}")
    if thought_changes:
        print(f"- Average thought change: {np.mean(thought_changes):.2f}")
        print(f"- Max thought change: {np.max(thought_changes):.2f}")
        print(f"- Min thought change: {np.min(thought_changes):.2f}")
    
    # Visualize
    os.makedirs("visualizations", exist_ok=True)
    visualize_simple(thought_norms, thought_changes, tokens, "visualizations/thought_evolution.png")


if __name__ == "__main__":
    main()