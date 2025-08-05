"""
Enhanced Diffusion Thought Tensor Model with Temporal Stacking
Implements thought accumulation over time in a fixed-size stack
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
try:
    from .base_model import SinusoidalTimeEmbedding, TransformerBlock
except ImportError:
    from base_model import SinusoidalTimeEmbedding, TransformerBlock


class LearnableTemporalPooling(nn.Module):
    """Attention-based pooling instead of simple mean pooling"""
    def __init__(self, hidden_dim, num_heads=1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
    
    def forward(self, x):
        # x: (batch, seq, hidden)
        batch_size = x.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        pooled, _ = self.attention(query, x, x)
        return pooled.squeeze(1)


class ThoughtStackEncoder(nn.Module):
    """Encodes a stack of thought tensors to embedding space - VECTORIZED VERSION"""
    
    def __init__(self, thought_dims, stack_size, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.thought_dims = thought_dims
        self.stack_size = stack_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # Each thought tensor gets flattened
        single_thought_size = math.prod(thought_dims)
        
        # Process each thought separately then combine (vectorized)
        self.thought_processor = nn.Sequential(
            nn.Linear(single_thought_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),  # Add dropout for regularization
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)   # Additional dropout
        )
        
        # Add temporal positional encoding
        self.temporal_pos_encoding = nn.Parameter(
            torch.randn(1, stack_size, hidden_dim) * 0.02
        )
        
        # Temporal attention over the stack  
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Learnable pooling instead of mean
        self.learnable_pool = LearnableTemporalPooling(hidden_dim)
        
        # Final projection with regularization
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, output_dim),
            nn.Dropout(dropout / 2)  # Lighter dropout before output
        )
    
    def forward(self, thought_stack):
        """
        Args:
            thought_stack: (batch_size, stack_size, *thought_dims)
        Returns:
            encoded: (batch_size, output_dim)
        """
        batch_size, stack_size = thought_stack.shape[:2]
        
        # Vectorized processing - process all thoughts at once
        thought_flat = thought_stack.view(batch_size * stack_size, -1)
        processed = self.thought_processor(thought_flat)
        processed = processed.view(batch_size, stack_size, -1)
        
        # Add temporal positional encoding
        processed = processed + self.temporal_pos_encoding
        
        # Apply temporal attention
        attended, _ = self.temporal_attention(processed, processed, processed)
        
        # Learnable pooling instead of mean
        pooled = self.learnable_pool(attended)
        
        # Final projection
        output = self.output_projection(pooled)
        
        return output


class ThoughtStackDecoder(nn.Module):
    """Decodes embeddings to a new thought tensor for the stack"""
    
    def __init__(self, input_dim, thought_dims, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.thought_dims = thought_dims
        self.dropout = dropout
        single_thought_size = math.prod(thought_dims)
        
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, single_thought_size),
            nn.Tanh()  # Bound output values
        )
    
    def forward(self, embeddings):
        batch_size = embeddings.shape[0]
        decoded = self.decoder(embeddings)
        return decoded.view(batch_size, *self.thought_dims)


class ThoughtStack:
    """Manages a fixed-size stack of thought tensors with FIFO behavior"""
    
    def __init__(self, stack_size, thought_dims, device='cuda'):
        self.stack_size = stack_size
        self.thought_dims = thought_dims
        self.device = device
        
    def initialize_stack(self, batch_size):
        """Initialize empty stack with zeros"""
        return torch.zeros(
            batch_size, self.stack_size, *self.thought_dims, 
            device=self.device
        )
    
    def push_thought(self, stack, new_thought):
        """
        Add new thought to stack, removing oldest if full
        Args:
            stack: (batch_size, stack_size, *thought_dims)
            new_thought: (batch_size, *thought_dims)
        Returns:
            updated_stack: (batch_size, stack_size, *thought_dims)
        """
        # Shift all thoughts by one position (remove oldest)
        shifted_stack = torch.cat([
            stack[:, 1:],  # Remove first (oldest) thought
            new_thought.unsqueeze(1)  # Add new thought at end
        ], dim=1)
        
        return shifted_stack
    
    def get_latest_thought(self, stack):
        """Get the most recent thought from stack"""
        return stack[:, -1]  # Last element is most recent
    
    def get_oldest_thought(self, stack):
        """Get the oldest thought from stack"""
        return stack[:, 0]  # First element is oldest


class EnhancedDiffusionThoughtModel(nn.Module):
    """Enhanced model with temporal thought stacking and gradient checkpointing"""
    
    def __init__(
        self,
        vocab_size=50257,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        max_seq_length=256,
        max_new_tokens=1024,  # Maximum tokens to generate in response
        thought_dims=(32, 32, 16),
        thought_stack_size=8,  # Number of thoughts to maintain
        dropout=0.1,
        thought_dropout=0.15,  # Specific dropout for thought layers
        noise_injection_std=0.01,  # Noise injection during training
        use_gradient_checkpointing=True
    ):
        super().__init__()
        
        # Model configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.thought_dims = thought_dims
        self.thought_stack_size = thought_stack_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.noise_injection_std = noise_injection_std
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_seq_length, embed_dim)
        
        # Thought stack processing with regularization
        self.thought_stack_encoder = ThoughtStackEncoder(
            thought_dims, 
            thought_stack_size, 
            hidden_dim=512, 
            output_dim=embed_dim,
            dropout=thought_dropout
        )
        self.thought_decoder = ThoughtStackDecoder(
            embed_dim, 
            thought_dims,
            hidden_dim=512,
            dropout=thought_dropout
        )
        
        # Stack manager
        self.thought_stack = ThoughtStack(thought_stack_size, thought_dims)
        
        # Time embedding for diffusion
        self.time_embed = SinusoidalTimeEmbedding(embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, noisy_tokens, thought_stack, timestep):
        """
        Args:
            noisy_tokens: (batch_size, seq_length)
            thought_stack: (batch_size, stack_size, *thought_dims)
            timestep: (batch_size,)
        Returns:
            logits: (batch_size, seq_length, vocab_size)
            new_thought: (batch_size, *thought_dims)
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
        
        # Apply transformer blocks with gradient checkpointing
        x = x.transpose(0, 1)  # (batch, seq, embed) -> (seq, batch, embed)
        for block in self.transformer_blocks:
            if self.training and self.use_gradient_checkpointing:
                x = checkpoint(
                    block, x, thought_embed, use_reentrant=False
                )
            else:
                x = block(x, thought_embed)
        x = x.transpose(0, 1)  # (seq, batch, embed) -> (batch, seq, embed)
        
        # Output processing
        x = self.output_norm(x)
        logits = self.output_projection(x)
        
        # Generate new thought tensor with noise injection during training
        # Use mean pooling over sequence for thought evolution
        pooled = x.mean(dim=1)
        new_thought = self.thought_decoder(pooled)
        
        # Add noise injection during training for regularization
        if self.training and self.noise_injection_std > 0:
            noise = torch.randn_like(new_thought) * self.noise_injection_std
            new_thought = new_thought + noise
        
        return logits, new_thought
    
    def evolve_thought_stack(self, thought_stack, new_thought):
        """Update the thought stack with a new thought"""
        return self.thought_stack.push_thought(thought_stack, new_thought)
    
    def initialize_thought_stack(self, batch_size, device):
        """Initialize a new thought stack"""
        self.thought_stack.device = device
        return self.thought_stack.initialize_stack(batch_size)
    
    def generate_with_stack(self, initial_stack=None, prompt_tokens=None, num_steps=500, temperature=1.0):
        """Generate text using diffusion process with thought stack"""
        batch_size = 1 if initial_stack is None else initial_stack.shape[0]
        device = next(self.parameters()).device
        
        # Initialize thought stack if not provided
        if initial_stack is None:
            thought_stack = self.initialize_thought_stack(batch_size, device)
        else:
            thought_stack = initial_stack
        
        if prompt_tokens is None:
            # Generate full sequence from scratch
            seq_length = self.max_seq_length
            tokens = torch.randint(0, self.vocab_size, (batch_size, seq_length), device=device)
            prompt_length = 0
        else:
            # Diffusion generation: fixed prompt + generated output
            prompt_length = prompt_tokens.shape[1]
            output_length = min(self.max_new_tokens, self.max_seq_length - prompt_length)
            
            if output_length <= 0:
                raise ValueError(f"Prompt too long ({prompt_length} tokens). Max sequence length is {self.max_seq_length}")
            
            # Start with random tokens for the output portion
            output_tokens = torch.randint(0, self.vocab_size, (batch_size, output_length), device=device)
            tokens = torch.cat([prompt_tokens, output_tokens], dim=1)
            seq_length = tokens.shape[1]
        
        # Track thought evolution
        thought_history = []
        
        # Diffusion sampling loop
        for t in reversed(range(num_steps)):
            timestep = torch.tensor([t], device=device).repeat(batch_size)
            
            # Forward pass
            logits, new_thought = self.forward(tokens, thought_stack, timestep)
            
            # Update thought stack
            thought_stack = self.evolve_thought_stack(thought_stack, new_thought)
            thought_history.append(new_thought.clone())
            
            # Sample new tokens - only modify output portion if we have a prompt
            if prompt_tokens is not None:
                # Keep prompt fixed, only denoise output tokens
                output_logits = logits[:, prompt_length:]
                
                if t > 0:
                    # Add noise for intermediate steps
                    probs = F.softmax(output_logits / temperature, dim=-1)
                    new_output_tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1)
                    new_output_tokens = new_output_tokens.view(batch_size, -1)
                else:
                    # Final step - deterministic
                    new_output_tokens = output_logits.argmax(dim=-1)
                
                # Reconstruct full sequence: fixed prompt + denoised output
                tokens = torch.cat([prompt_tokens, new_output_tokens], dim=1)
            else:
                # No prompt - denoise entire sequence
                if t > 0:
                    probs = F.softmax(logits / temperature, dim=-1)
                    tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1)
                    tokens = tokens.view(batch_size, seq_length)
                else:
                    tokens = logits.argmax(dim=-1)
        
        return tokens, thought_stack, thought_history


def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test enhanced model creation
    model = EnhancedDiffusionThoughtModel(
        vocab_size=1000,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        max_seq_length=64,
        thought_dims=(8, 8, 4),
        thought_stack_size=8
    )
    
    param_count = count_parameters(model)
    print(f"Enhanced model created with {param_count / 1e6:.2f}M parameters")
    
    # Test forward pass
    batch_size = 2
    seq_length = 32
    
    dummy_tokens = torch.randint(0, 1000, (batch_size, seq_length))
    dummy_stack = model.initialize_thought_stack(batch_size, 'cpu')
    dummy_timestep = torch.tensor([250, 250])
    
    with torch.no_grad():
        logits, new_thought = model(dummy_tokens, dummy_stack, dummy_timestep)
        updated_stack = model.evolve_thought_stack(dummy_stack, new_thought)
    
    print(f"Output shapes:")
    print(f"  Logits: {logits.shape}")
    print(f"  New thought: {new_thought.shape}")
    print(f"  Updated stack: {updated_stack.shape}")
    print("✅ Enhanced model test passed!")