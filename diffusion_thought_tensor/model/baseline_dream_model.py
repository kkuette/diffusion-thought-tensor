"""
Baseline DREAM Model - Identical to StackedDiffusionModel3D but WITHOUT thought system
This serves as the control group to test if thoughts actually contribute to performance.

Key differences from stacked_3d_model.py:
- NO ThoughtStack3D
- NO seq_to_thought/thought_to_seq projections  
- NO cross-attention with thoughts
- NO generate_new_thought method
- Same architecture otherwise (embed_dim, layers, masking, DREAM features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class MaskingStrategy:
    """Configuration for token masking strategies"""
    mask_ratio: float = 0.5  # Ratio of tokens to mask initially
    confidence_threshold: float = 0.8  # Threshold for unmasking tokens
    progressive_unmasking: bool = True  # Whether to progressively unmask
    block_size: int = 1  # Size of blocks to unmask together
    mask_token_id: int = -1  # Special token ID for masked positions


class BidirectionalMaskedAttention(nn.Module):
    """Bidirectional attention with support for token masking (DREAM-style)"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, max_seq_length=2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # No causal mask for bidirectional attention
        self.max_seq_length = max_seq_length
    
    def forward(self, x, attention_mask=None, token_mask=None):
        """
        Forward pass with bidirectional attention and token masking
        Args:
            x: input tensor (batch, seq_len, embed_dim)
            attention_mask: attention mask (batch, seq_len, seq_len) 
            token_mask: token-level mask (batch, seq_len) - True for masked tokens
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, embed_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: (batch, num_heads, seq_len, head_dim)
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        # scores: (batch, num_heads, seq_len, seq_len)
        
        # No causal mask for bidirectional attention (DREAM-style)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Expand to match scores dimensions
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply token masking - for bidirectional attention, masked tokens can see all positions
        # This allows better context understanding during diffusion
        if token_mask is not None:
            # In bidirectional setting, we don't restrict what masked tokens can see
            # This is key to DREAM's improved performance
            pass  # No additional masking needed
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        # attn_output: (batch, num_heads, seq_len, head_dim)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # Final projection
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output, attn_weights


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embeddings for diffusion timesteps"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class BaselineDreamModel(nn.Module):
    """
    Baseline DREAM model without thought system.
    Identical to StackedDiffusionModel3D except:
    - NO thought stack
    - NO thought projections
    - NO cross-attention with thoughts
    
    This is the control group for testing thought contribution.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        embed_dim: int = 768,
        num_layers: int = 8,
        num_heads: int = 12,
        max_seq_length: int = 256,
        dropout: float = 0.1,
        masking_strategy: Optional[MaskingStrategy] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        
        # Default masking strategy with DREAM settings
        if masking_strategy is None:
            masking_strategy = MaskingStrategy(
                mask_ratio=0.7,  # DREAM-style masking ratio
                confidence_threshold=0.75,
                progressive_unmasking=True
            )
        self.masking_strategy = masking_strategy

        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_length, embed_dim)
        
        # Timestep embedding for diffusion
        self.timestep_embed = SinusoidalTimeEmbedding(embed_dim)
        self.timestep_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # DREAM-style transformer layers with bidirectional attention
        # NOTE: NO cross-attention with thoughts - that's the key difference
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "self_attn": BidirectionalMaskedAttention(
                            embed_dim, num_heads, dropout, max_seq_length
                        ),
                        # NO cross_attn with thoughts - this is the baseline!
                        "norm1": nn.LayerNorm(embed_dim),
                        "norm2": nn.LayerNorm(embed_dim),
                        "mlp": nn.Sequential(
                            nn.Linear(embed_dim, embed_dim * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(embed_dim * 4, embed_dim),
                            nn.Dropout(dropout),
                        ),
                    }
                )
            )

        # Output
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Baseline forward pass WITHOUT thought system
        Args:
            tokens: (B, seq_len)
            timestep: Optional diffusion timestep (B,)
            token_mask: Optional token mask (B, seq_len) - True for masked tokens
        Returns:
            logits: (B, seq_len, vocab_size)
        """
        B, seq_len = tokens.shape
        device = tokens.device

        # Embed tokens
        x = self.token_embed(tokens)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = x + self.pos_embed(positions)

        # Add timestep embedding if provided
        if timestep is not None:
            time_embed = self.timestep_embed(timestep)  # (B, embed_dim)
            time_embed = self.timestep_proj(time_embed)  # (B, embed_dim)
            x = x + time_embed.unsqueeze(1) * 0.1  # Add to all positions

        # NO thought influence - that's the key difference!

        # Process through transformer layers
        for layer in self.layers:
            # Self-attention with token masking support
            attn_out, _ = layer["self_attn"](layer["norm1"](x), token_mask=token_mask)
            x = x + attn_out

            # NO cross-attention with thoughts!
            
            # MLP
            x = x + layer["mlp"](layer["norm2"](x))

        # Output projection
        x = self.output_norm(x)
        logits = self.output_proj(x)

        return logits


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create baseline model with same config as thought model
    model = BaselineDreamModel(
        vocab_size=50257,
        embed_dim=720,
        num_layers=16,
        num_heads=16,
        max_seq_length=2048,
        dropout=0.1,
    ).to(device)

    print(f"✅ Baseline DREAM Model created")
    print(f"   Parameters: {count_parameters(model) / 1e6:.2f}M")
    print(f"   Embed dim: {model.embed_dim}")
    print(f"   Layers: {len(model.layers)}")
    print(f"   Max seq length: {model.max_seq_length}")

    # Test forward pass
    batch_size = 2
    seq_length = 64

    tokens = torch.randint(0, 50257, (batch_size, seq_length)).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    token_mask = torch.rand(batch_size, seq_length) < 0.3  # 30% masking
    token_mask = token_mask.to(device)

    print(f"\n📥 Input shapes:")
    print(f"  Tokens: {tokens.shape}")
    print(f"  Timesteps: {timesteps.shape}")
    print(f"  Token mask ratio: {token_mask.float().mean():.2f}")

    # Forward pass
    logits = model(tokens, timesteps, token_mask)

    print(f"\n📤 Output shapes:")
    print(f"  Logits: {logits.shape}")
    print(f"  Logits range: [{logits.min():.3f}, {logits.max():.3f}]")

    print("\n✅ Baseline model test passed!")
    print("   Ready for controlled comparison with thought model")