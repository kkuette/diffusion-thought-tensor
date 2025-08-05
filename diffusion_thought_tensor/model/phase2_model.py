"""
Phase 2 Diffusion Thought Tensor Model - Scaled to ~500M Parameters
Optimized for RTX 3090 with advanced features and memory efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import yaml
from typing import Optional, Tuple, Dict, Any

try:
    from .enhanced_model import ThoughtStack, SinusoidalTimeEmbedding
    from .base_model import TransformerBlock
except ImportError:
    from enhanced_model import ThoughtStack, SinusoidalTimeEmbedding
    from base_model import TransformerBlock


class ScaledThoughtStackEncoder(nn.Module):
    """Scaled encoder for thought stacks with larger capacity"""
    
    def __init__(self, thought_dims, stack_size, hidden_dim, output_dim, num_attention_layers=3):
        super().__init__()
        self.thought_dims = thought_dims
        self.stack_size = stack_size
        self.hidden_dim = hidden_dim
        
        # Each thought tensor gets flattened
        single_thought_size = math.prod(thought_dims)
        
        # Multi-layer thought processor
        self.thought_processor = nn.Sequential(
            nn.Linear(single_thought_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Multi-layer temporal attention
        self.temporal_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                hidden_dim, num_heads=16, dropout=0.1, batch_first=True
            ) for _ in range(num_attention_layers)
        ])
        
        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_attention_layers)
        ])
        
        # Enhanced feed-forward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 4, hidden_dim)
            ) for _ in range(num_attention_layers)
        ])
        
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_attention_layers)
        ])
        
        # Enhanced output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )
    
    def forward(self, thought_stack):
        """
        Args:
            thought_stack: (batch_size, stack_size, *thought_dims)
        Returns:
            encoded: (batch_size, output_dim)
        """
        batch_size, stack_size = thought_stack.shape[:2]
        
        # Flatten each thought in the stack
        thought_flat = thought_stack.view(batch_size, stack_size, -1)
        
        # Process each thought separately
        processed = []
        for i in range(stack_size):
            thought_embed = self.thought_processor(thought_flat[:, i])
            processed.append(thought_embed)
        
        # Stack for temporal attention
        x = torch.stack(processed, dim=1)  # (batch, stack_size, hidden)
        
        # Apply multiple temporal attention layers
        for attn, attn_norm, ffn, ffn_norm in zip(
            self.temporal_attention_layers, self.attention_norms, 
            self.ffns, self.ffn_norms
        ):
            # Self-attention
            attn_out, _ = attn(x, x, x)
            x = attn_norm(x + attn_out)
            
            # Feed-forward
            ffn_out = ffn(x)
            x = ffn_norm(x + ffn_out)
        
        # Pool over time dimension (learnable pooling could be added)
        pooled = x.mean(dim=1)  # (batch, hidden)
        
        # Final projection
        output = self.output_projection(pooled)
        
        return output


class ScaledThoughtStackDecoder(nn.Module):
    """Scaled decoder for generating new thought tensors"""
    
    def __init__(self, input_dim, thought_dims, hidden_dim=1024):
        super().__init__()
        self.thought_dims = thought_dims
        single_thought_size = math.prod(thought_dims)
        
        # Multi-layer decoder with residual connections
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(4)
        ])
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, single_thought_size),
            nn.Tanh()  # Bound output values
        )
    
    def forward(self, embeddings):
        batch_size = embeddings.shape[0]
        
        # Input projection
        x = self.input_norm(self.input_projection(embeddings))
        
        # Multi-layer processing with residuals
        for layer in self.decoder_layers:
            x = x + layer(x)
        
        # Output projection
        decoded = self.output_projection(x)
        
        return decoded.view(batch_size, *self.thought_dims)


class ScaledTransformerBlock(nn.Module):
    """Enhanced transformer block optimized for large models"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_glu=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_glu = use_glu
        
        # Self-attention for text
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        
        # Cross-attention with thought tensor
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attn_norm = nn.LayerNorm(embed_dim)
        
        # Enhanced feed-forward network with optional GLU
        if use_glu:
            # GLU variant for better performance
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 8),  # Larger expansion
                nn.GLU(dim=-1),  # Gated Linear Unit
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim)
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim)
            )
        
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, thought_embed=None):
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.self_attn_norm(x + self.dropout(attn_out))
        
        # Cross-attention with thought (if provided)
        if thought_embed is not None:
            seq_len, batch_size, embed_dim = x.shape
            thought_embed_expanded = thought_embed.unsqueeze(0).expand(1, batch_size, embed_dim)
            cross_out, _ = self.cross_attn(x, thought_embed_expanded, thought_embed_expanded)
            x = self.cross_attn_norm(x + self.dropout(cross_out))
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + self.dropout(ffn_out))
        
        return x


class Phase2DiffusionThoughtModel(nn.Module):
    """Phase 2 model scaled to ~500M parameters with advanced features"""
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        super().__init__()
        
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = self._default_config()
        
        # Override with any provided kwargs
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested config like 'model.embed_dim'
                section, param = key.split('.', 1)
                if section not in config:
                    config[section] = {}
                config[section][param] = value
            else:
                config[key] = value
        
        self.config = config
        model_config = config['model']
        thought_config = config['thought_tensor']
        
        # Model configuration
        self.vocab_size = model_config['vocab_size']
        self.embed_dim = model_config['embed_dim']
        self.num_layers = model_config['num_layers']
        self.num_heads = model_config['num_heads']
        self.max_seq_length = model_config['max_seq_length']
        self.thought_dims = tuple(thought_config['input_dims'])
        self.thought_stack_size = thought_config['stack_size']
        
        # Token embeddings (largest component)
        self.token_embeddings = nn.Embedding(self.vocab_size, self.embed_dim)
        self.position_embeddings = nn.Embedding(self.max_seq_length, self.embed_dim)
        
        # Scaled thought stack processing
        self.thought_stack_encoder = ScaledThoughtStackEncoder(
            self.thought_dims,
            self.thought_stack_size,
            hidden_dim=thought_config['hidden_dim'],
            output_dim=self.embed_dim,
            num_attention_layers=4  # More layers for complex processing
        )
        
        self.thought_decoder = ScaledThoughtStackDecoder(
            self.embed_dim,
            self.thought_dims,
            hidden_dim=thought_config['hidden_dim']
        )
        
        # Stack manager
        self.thought_stack = ThoughtStack(self.thought_stack_size, self.thought_dims)
        
        # Enhanced time embedding
        self.time_embed = SinusoidalTimeEmbedding(self.embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim * 4, self.embed_dim * 4),
            nn.GELU(),
            nn.Linear(self.embed_dim * 4, self.embed_dim)
        )
        
        # Scaled transformer blocks
        self.transformer_blocks = nn.ModuleList([
            ScaledTransformerBlock(
                self.embed_dim, 
                self.num_heads, 
                dropout=model_config['dropout'],
                use_glu=True
            ) for _ in range(self.num_layers)
        ])
        
        # Output layers
        self.output_norm = nn.LayerNorm(self.embed_dim)
        self.output_projection = nn.Linear(self.embed_dim, self.vocab_size)
        
        # Enable gradient checkpointing if specified
        if config.get('hardware', {}).get('gradient_checkpointing', False):
            self.gradient_checkpointing = True
        else:
            self.gradient_checkpointing = False
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply parameter-efficient initialization for large models
        self._scale_initialization()
    
    def _default_config(self):
        """Default configuration for the model"""
        return {
            'model': {
                'embed_dim': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'vocab_size': 50257,
                'max_seq_length': 512,
                'dropout': 0.1
            },
            'thought_tensor': {
                'input_dims': [32, 32, 16],
                'stack_size': 16,
                'hidden_dim': 1024
            },
            'hardware': {
                'gradient_checkpointing': True
            }
        }
    
    def _init_weights(self, module):
        """Initialize weights with proper scaling for large models"""
        if isinstance(module, nn.Linear):
            # Use scaled initialization
            std = 0.02 / math.sqrt(2 * self.num_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def _scale_initialization(self):
        """Apply additional scaling for large model stability"""
        # Scale down the output projection for stability
        with torch.no_grad():
            self.output_projection.weight.mul_(0.5)
            
        # Scale down attention weights in deeper layers
        for i, block in enumerate(self.transformer_blocks):
            layer_scale = 1.0 / math.sqrt(i + 1)
            with torch.no_grad():
                if hasattr(block.self_attn, 'out_proj'):
                    block.self_attn.out_proj.weight.mul_(layer_scale)
                if hasattr(block.cross_attn, 'out_proj'):
                    block.cross_attn.out_proj.weight.mul_(layer_scale)
    
    def forward(self, noisy_tokens, thought_stack, timestep):
        """
        Forward pass with optional gradient checkpointing
        """
        batch_size, seq_length = noisy_tokens.shape
        device = noisy_tokens.device
        
        # Get embeddings
        token_embeds = self.token_embeddings(noisy_tokens)
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0)
        position_embeds = self.position_embeddings(position_ids)
        
        # Add position embeddings
        x = token_embeds + position_embeds
        
        # Encode thought stack
        thought_embed = self.thought_stack_encoder(thought_stack)
        
        # Add time embedding
        time_embed = self.time_embed(timestep)
        time_embed = self.time_mlp(time_embed)
        x = x + time_embed.unsqueeze(1)
        
        # Apply transformer blocks (convert to seq_first format)
        x = x.transpose(0, 1)  # (batch, seq, embed) -> (seq, batch, embed)
        
        if self.gradient_checkpointing and self.training:
            # Use gradient checkpointing for memory efficiency
            from torch.utils.checkpoint import checkpoint
            for block in self.transformer_blocks:
                x = checkpoint(block, x, thought_embed, use_reentrant=False)
        else:
            for block in self.transformer_blocks:
                x = block(x, thought_embed)
        
        x = x.transpose(0, 1)  # (seq, batch, embed) -> (batch, seq, embed)
        
        # Output processing
        x = self.output_norm(x)
        logits = self.output_projection(x)
        
        # Generate new thought tensor
        pooled = x.mean(dim=1)
        new_thought = self.thought_decoder(pooled)
        
        return logits, new_thought
    
    def evolve_thought_stack(self, thought_stack, new_thought):
        """Update the thought stack with a new thought"""
        return self.thought_stack.push_thought(thought_stack, new_thought)
    
    def initialize_thought_stack(self, batch_size, device):
        """Initialize a new thought stack"""
        self.thought_stack.device = device
        return self.thought_stack.initialize_stack(batch_size)
    
    def get_memory_stats(self):
        """Get memory usage statistics"""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
            }
        return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}


def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_phase2_model(config_path: str, checkpoint_path: Optional[str] = None):
    """Load Phase 2 model with optional checkpoint"""
    model = Phase2DiffusionThoughtModel(config_path=config_path)
    
    if checkpoint_path and torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    return model


if __name__ == "__main__":
    # Test Phase 2 model creation
    print("Creating Phase 2 model...")
    
    # Test with default configuration
    model = Phase2DiffusionThoughtModel()
    param_count = count_parameters(model)
    print(f"Phase 2 model created with {param_count / 1e6:.1f}M parameters")
    
    # Test forward pass
    batch_size = 1  # Small batch for testing
    seq_length = 64
    
    dummy_tokens = torch.randint(0, model.vocab_size, (batch_size, seq_length))
    dummy_stack = model.initialize_thought_stack(batch_size, 'cpu')
    dummy_timestep = torch.tensor([500])
    
    print(f"Input shapes:")
    print(f"  Tokens: {dummy_tokens.shape}")
    print(f"  Stack: {dummy_stack.shape}")
    print(f"  Timestep: {dummy_timestep.shape}")
    
    with torch.no_grad():
        logits, new_thought = model(dummy_tokens, dummy_stack, dummy_timestep)
        updated_stack = model.evolve_thought_stack(dummy_stack, new_thought)
    
    print(f"Output shapes:")
    print(f"  Logits: {logits.shape}")
    print(f"  New thought: {new_thought.shape}")
    print(f"  Updated stack: {updated_stack.shape}")
    
    # Memory statistics
    memory_stats = model.get_memory_stats()
    print(f"Memory usage: {memory_stats}")
    
    print("✅ Phase 2 model test passed!")