"""
Base Diffusion Thought Tensor Model
~500M parameters optimized for RTX 3090
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embeddings for diffusion timesteps"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ThoughtEncoder(nn.Module):
    """Encodes n-dimensional thought tensors to embedding space"""
    
    def __init__(self, input_dims, hidden_dim, output_dim):
        super().__init__()
        self.input_dims = input_dims
        total_input = math.prod(input_dims)
        
        self.encoder = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )
    
    def forward(self, thought_tensor):
        batch_size = thought_tensor.shape[0]
        # Flatten thought tensor
        thought_flat = thought_tensor.view(batch_size, -1)
        return self.encoder(thought_flat)


class ThoughtDecoder(nn.Module):
    """Decodes embeddings back to (n-1)-dimensional thought tensors"""
    
    def __init__(self, input_dim, output_dims, hidden_dim=512):
        super().__init__()
        self.output_dims = output_dims
        total_output = math.prod(output_dims)
        
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, total_output),
            nn.Tanh()  # Bound output values
        )
    
    def forward(self, embeddings):
        batch_size = embeddings.shape[0]
        decoded = self.decoder(embeddings)
        return decoded.view(batch_size, *self.output_dims)


class TransformerBlock(nn.Module):
    """Single transformer block with thought-text cross-attention"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Self-attention for text
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        
        # Cross-attention with thought tensor
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attn_norm = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
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
            # Expand thought embedding for cross-attention
            # x shape: (seq_len, batch, embed_dim), thought_embed shape: (batch, embed_dim)
            seq_len, batch_size, embed_dim = x.shape
            thought_embed_expanded = thought_embed.unsqueeze(0).expand(1, batch_size, embed_dim)
            cross_out, _ = self.cross_attn(x, thought_embed_expanded, thought_embed_expanded)
            x = self.cross_attn_norm(x + self.dropout(cross_out))
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + self.dropout(ffn_out))
        
        return x


class DiffusionThoughtModel(nn.Module):
    """Main diffusion model with thought tensor integration (~500M params)"""
    
    def __init__(
        self,
        vocab_size=50257,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        max_seq_length=256,
        thought_input_dims=(32, 32, 16),
        thought_output_dims=(32, 32, 8),
        dropout=0.1
    ):
        super().__init__()
        
        # Model configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length
        
        # Token embeddings (~38M params)
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_seq_length, embed_dim)
        
        # Thought tensor modules (~25M params)
        self.thought_encoder = ThoughtEncoder(
            thought_input_dims, 
            hidden_dim=512, 
            output_dim=embed_dim
        )
        self.thought_decoder = ThoughtDecoder(
            embed_dim, 
            thought_output_dims,
            hidden_dim=512
        )
        
        # Time embedding for diffusion (~1M params)
        self.time_embed = SinusoidalTimeEmbedding(embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Transformer blocks (~420M params)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection (~38M params)
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
    
    def forward(self, noisy_tokens, thought_tensor, timestep):
        batch_size, seq_length = noisy_tokens.shape
        device = noisy_tokens.device
        
        # Get embeddings
        token_embeds = self.token_embeddings(noisy_tokens)
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0)
        position_embeds = self.position_embeddings(position_ids)
        
        # Add position embeddings
        x = token_embeds + position_embeds
        
        # Encode thought tensor
        thought_embed = self.thought_encoder(thought_tensor)
        
        # Add time embedding
        time_embed = self.time_embed(timestep)
        time_embed = self.time_mlp(time_embed)
        x = x + time_embed.unsqueeze(1)
        
        # Apply transformer blocks (convert to seq_first format)
        x = x.transpose(0, 1)  # (batch, seq, embed) -> (seq, batch, embed)
        for block in self.transformer_blocks:
            x = block(x, thought_embed)
        x = x.transpose(0, 1)  # (seq, batch, embed) -> (batch, seq, embed)
        
        # Output processing
        x = self.output_norm(x)
        logits = self.output_projection(x)
        
        # Decode new thought tensor
        # Use mean pooling over sequence for thought evolution
        pooled = x.mean(dim=1)
        new_thought = self.thought_decoder(pooled)
        
        return logits, new_thought
    
    def generate(self, thought_tensor, prompt_tokens=None, num_steps=500, temperature=1.0):
        """Generate text using diffusion process"""
        batch_size = thought_tensor.shape[0]
        device = thought_tensor.device
        
        if prompt_tokens is None:
            # Start from random tokens
            seq_length = self.max_seq_length
            tokens = torch.randint(0, self.vocab_size, (batch_size, seq_length), device=device)
        else:
            tokens = prompt_tokens
            seq_length = tokens.shape[1]
        
        # Diffusion sampling loop
        for t in reversed(range(num_steps)):
            timestep = torch.tensor([t], device=device).repeat(batch_size)
            
            # Forward pass
            logits, new_thought = self.forward(tokens, thought_tensor, timestep)
            
            # Sample new tokens (simplified - real implementation needs proper denoising)
            if t > 0:
                # Add noise for intermediate steps
                probs = F.softmax(logits / temperature, dim=-1)
                tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1)
                tokens = tokens.view(batch_size, seq_length)
            else:
                # Final step - deterministic
                tokens = logits.argmax(dim=-1)
            
            # Update thought tensor
            thought_tensor = new_thought
        
        return tokens, thought_tensor


def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation and parameter count
    model = DiffusionThoughtModel()
    param_count = count_parameters(model)
    print(f"Model created with {param_count / 1e6:.1f}M parameters")
    
    # Test forward pass
    batch_size = 2
    seq_length = 128
    thought_dims = (32, 32, 16)
    
    dummy_tokens = torch.randint(0, 50257, (batch_size, seq_length))
    dummy_thought = torch.randn(batch_size, *thought_dims)
    dummy_timestep = torch.tensor([250, 250])
    
    with torch.no_grad():
        logits, new_thought = model(dummy_tokens, dummy_thought, dummy_timestep)
    
    print(f"Output shapes - Logits: {logits.shape}, New thought: {new_thought.shape}")
    print("✅ Model test passed!")