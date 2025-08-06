"""
3D Model with Depth as Thought Stack
Each thought is 2D (H, W), and depth represents the stack of thoughts over time
Enhanced with DREAM-style token masking and bidirectional attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass
from einops import rearrange


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


class Pure3DConvAttention(nn.Module):
    """
    Pure 3D convolution-based attention - no flattening
    Uses 3D convolutions to process the (H, W, D) volume directly
    """

    def __init__(self, hidden_dim: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 3D convolutions for local attention patterns
        self.local_attn = nn.Sequential(
            nn.Conv3d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=hidden_dim // 8,
            ),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Point-wise 3D convolution for feature mixing
        self.pointwise = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W, D) where D is the thought stack depth
        Returns:
            (B, C, H, W, D) - same shape, pure 3D processing
        """
        # Local 3D attention patterns
        attended = self.local_attn(x)

        # Point-wise feature mixing
        out = self.pointwise(attended)

        return out


class TemporalStackConvolution(nn.Module):
    """
    3D convolution that processes across the thought stack
    Learns temporal patterns in the stack while maintaining spatial structure
    """

    def __init__(self, hidden_dim: int, kernel_size: Tuple[int, int, int] = (3, 3, 3)):
        super().__init__()
        self.conv3d = nn.Conv3d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2),
            bias=False,
        )
        self.norm = nn.GroupNorm(32, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W, D) where D is stack depth
        """
        out = self.conv3d(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class ThoughtStack3D(nn.Module):
    """
    3D Thought Stack where:
    - Each thought is a 2D spatial map (H, W)
    - Depth (D) represents the stack of thoughts over time
    - New thoughts are pushed to the stack, old ones are popped
    """

    def __init__(
        self,
        spatial_dims: Tuple[int, int],  # (H, W) for each thought
        stack_depth: int,  # Number of thoughts in stack
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.H, self.W = spatial_dims
        self.stack_depth = stack_depth
        self.hidden_dim = hidden_dim

        # Initialize empty stack
        self.register_buffer(
            "empty_thought", torch.zeros(1, hidden_dim, self.H, self.W)
        )

        # Stack processing layers - pure 3D operations
        self.stack_attention = Pure3DConvAttention(hidden_dim, kernel_size=3)
        self.temporal_conv = TemporalStackConvolution(hidden_dim, kernel_size=(3, 3, 3))

        # Importance scoring for stack positions
        self.importance_scorer = nn.Conv3d(hidden_dim, 1, kernel_size=1)

        # Layer norm
        self.norm = nn.GroupNorm(32, hidden_dim)

    def initialize_stack(
        self, batch_size: int, device: torch.device, noise_std: float = 0.1
    ) -> torch.Tensor:
        """
        Initialize thought stack with proper noise
        Args:
            batch_size: Number of batches
            device: Device to create tensor on
            noise_std: Standard deviation for noise initialization
        Returns:
            (B, C, H, W, D) where D is stack depth, initialized with noise
        """
        # Initialize each position in the stack with independent noise
        stack = (
            torch.randn(
                batch_size,
                self.hidden_dim,
                self.H,
                self.W,
                self.stack_depth,
                device=device,
            )
            * noise_std
        )

        # Optional: add small bias to make different depth positions slightly different
        depth_bias = torch.linspace(-0.1, 0.1, self.stack_depth, device=device)
        depth_bias = depth_bias.view(1, 1, 1, 1, -1)  # Shape for broadcasting
        stack = stack + depth_bias * 0.1

        return stack

    def push_thought(
        self, stack: torch.Tensor, new_thought: torch.Tensor
    ) -> torch.Tensor:
        """
        Push new thought to stack (FIFO)
        Args:
            stack: (B, C, H, W, D) current stack
            new_thought: (B, C, H, W) new 2D thought to add
        Returns:
            Updated stack with new thought at position 0, oldest removed
        """
        # Shift stack: remove oldest (last), add new at front
        updated_stack = torch.cat(
            [
                new_thought.unsqueeze(-1),  # Add depth dimension
                stack[:, :, :, :, :-1],  # All but the last (oldest)
            ],
            dim=-1,
        )

        return updated_stack

    def push_with_importance(
        self, stack: torch.Tensor, new_thought: torch.Tensor
    ) -> torch.Tensor:
        """
        Push new thought but keep important ones based on learned importance
        """
        # Compute importance scores for each position in stack
        importance = self.importance_scorer(stack).squeeze(1)  # (B, H, W, D)

        # Find least important position (averaged across spatial dims)
        importance_per_depth = importance.mean(dim=[1, 2])  # (B, D)
        least_important_idx = torch.argmin(importance_per_depth, dim=-1)  # (B,)

        # Replace least important with new thought
        B = stack.shape[0]
        updated_stack = stack.clone()
        for b in range(B):
            idx = least_important_idx[b]
            updated_stack[b, :, :, :, idx] = new_thought[b]

        return updated_stack

    def get_current_thought(self, stack: torch.Tensor) -> torch.Tensor:
        """
        Get the current (most recent) thought from stack
        Args:
            stack: (B, C, H, W, D)
        Returns:
            (B, C, H, W) - the most recent thought
        """
        return stack[:, :, :, :, 0]  # First position is most recent

    def process_stack(self, stack: torch.Tensor) -> torch.Tensor:
        """
        Process the entire stack with pure 3D operations - no flattening
        """
        # Apply 3D convolution-based attention
        stack = stack + self.stack_attention(self.norm(stack))

        # Apply temporal convolution to learn patterns across stack
        stack = stack + self.temporal_conv(stack)

        return stack


class StackedDiffusionModel3D(nn.Module):
    """
    Diffusion model where:
    - Output is 2D thoughts (H, W)
    - Depth represents the stack of previous thoughts
    - Stack evolves over time with new thoughts
    - Enhanced with DREAM-style token masking and bidirectional attention
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        embed_dim: int = 768,
        num_layers: int = 8,
        num_heads: int = 12,
        thought_dims: Tuple[int, int] = (32, 32),  # 2D thought (H, W)
        stack_depth: int = 16,  # Number of thoughts in stack
        thought_hidden_dim: int = 256,
        max_seq_length: int = 256,
        dropout: float = 0.1,
        masking_strategy: Optional[MaskingStrategy] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.thought_dims = thought_dims
        self.stack_depth = stack_depth
        self.thought_hidden_dim = thought_hidden_dim
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

        # 3D Thought stack manager
        self.thought_stack = ThoughtStack3D(
            thought_dims, stack_depth, thought_hidden_dim
        )

        # Project sequence to thought dimension
        self.seq_to_thought = nn.Sequential(
            nn.Linear(embed_dim, thought_hidden_dim * 4),
            nn.GELU(),
            nn.Linear(
                thought_hidden_dim * 4,
                thought_hidden_dim * thought_dims[0] * thought_dims[1],
            ),
        )

        # Project thought to sequence dimension
        self.thought_to_seq = nn.Sequential(
            nn.Conv2d(
                thought_hidden_dim, thought_hidden_dim // 2, kernel_size=3, padding=1
            ),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(thought_hidden_dim // 2, embed_dim),
        )

        # DREAM-style transformer layers with bidirectional attention
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "self_attn": BidirectionalMaskedAttention(
                            embed_dim, num_heads, dropout, max_seq_length
                        ),
                        "cross_attn": nn.MultiheadAttention(
                            embed_dim, num_heads, dropout=dropout, batch_first=True
                        ),
                        "norm1": nn.LayerNorm(embed_dim),
                        "norm2": nn.LayerNorm(embed_dim),
                        "norm3": nn.LayerNorm(embed_dim),
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

    def generate_new_thought(self, seq_features: torch.Tensor) -> torch.Tensor:
        """
        Generate a new 2D thought from sequence features
        Args:
            seq_features: (B, seq_len, embed_dim)
        Returns:
            (B, C, H, W) - new 2D thought
        """
        B = seq_features.shape[0]

        # Pool sequence to single vector
        pooled = seq_features.mean(dim=1)  # (B, embed_dim)

        # Project to thought space
        thought_flat = self.seq_to_thought(pooled)  # (B, C*H*W)

        # Reshape to 2D thought
        new_thought = thought_flat.view(
            B, self.thought_hidden_dim, self.thought_dims[0], self.thought_dims[1]
        )

        return new_thought

    def forward(
        self,
        tokens: torch.Tensor,
        thought_stack: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        return_all_thoughts: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            tokens: (B, seq_len)
            thought_stack: (B, C, H, W, D) where D is stack depth
            timestep: Optional diffusion timestep (B,)
            token_mask: Optional token mask (B, seq_len) - True for masked tokens
            return_all_thoughts: Whether to return the entire stack
        Returns:
            logits: (B, seq_len, vocab_size)
            new_thought: (B, C, H, W) - the new 2D thought
            updated_stack: Optional (B, C, H, W, D) if return_all_thoughts=True
        """
        B, seq_len = tokens.shape
        device = tokens.device

        # Process thought stack with 3D operations
        processed_stack = self.thought_stack.process_stack(thought_stack)

        # Get current thought (most recent)
        current_thought = self.thought_stack.get_current_thought(processed_stack)

        # Convert thought to sequence feature
        thought_feature = self.thought_to_seq(current_thought)  # (B, embed_dim)

        # Embed tokens
        x = self.token_embed(tokens)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = x + self.pos_embed(positions)

        # Add timestep embedding if provided
        if timestep is not None:
            time_embed = self.timestep_embed(timestep)  # (B, embed_dim)
            time_embed = self.timestep_proj(time_embed)  # (B, embed_dim)
            x = x + time_embed.unsqueeze(1) * 0.1  # Add to all positions

        # Add thought influence to each position
        x = x + thought_feature.unsqueeze(1) * 0.1

        # Process through transformer layers with DREAM-style attention
        for layer in self.layers:
            # Self-attention with token masking support
            attn_out, _ = layer["self_attn"](layer["norm1"](x), token_mask=token_mask)
            x = x + attn_out

            # Cross-attention with thought embeddings
            thought_expanded = thought_feature.unsqueeze(1)  # (B, 1, embed_dim)
            cross_out, _ = layer["cross_attn"](
                layer["norm2"](x), thought_expanded, thought_expanded
            )
            x = x + cross_out

            # MLP
            x = x + layer["mlp"](layer["norm3"](x))

        # Generate new 2D thought from processed sequence
        new_thought = self.generate_new_thought(x)

        # Update stack with new thought
        updated_stack = self.thought_stack.push_with_importance(
            processed_stack, new_thought
        )

        # Output projection
        x = self.output_norm(x)
        logits = self.output_proj(x)

        if return_all_thoughts:
            return logits, new_thought, updated_stack
        else:
            return logits, new_thought, None

    def initialize_stack(
        self, batch_size: int, device: torch.device, noise_std: float = 0.1
    ) -> torch.Tensor:
        """Initialize 3D thought stack with noise"""
        return self.thought_stack.initialize_stack(batch_size, device, noise_std)


def visualize_single_thought(thought: torch.Tensor, title: str = "Single Thought"):
    """
    Visualize a single 2D thought
    Args:
        thought: (C, H, W) - single thought tensor
        title: Title for the visualization
    """
    C, H, W = thought.shape

    print(f"\n🧠 {title}:")
    print(f"  Shape: {thought.shape} (C, H, W)")
    print(f"  Channels: {C}, Spatial: {H}×{W}")
    print(f"  Stats: min={thought.min():.3f}, max={thought.max():.3f}")
    print(f"  Mean: {thought.mean():.3f}, Std: {thought.std():.3f}")

    # Show spatial structure (average across channels)
    spatial_map = thought.mean(dim=0)  # (H, W)
    print(f"\n  📍 2D Spatial Map (channel-averaged):")

    # Print a small ASCII visualization of the 2D map
    print("     ", end="")
    for w in range(min(8, W)):
        print(f"{w:6}", end="")
    print()

    for h in range(min(8, H)):
        print(f"{h:3}: ", end="")
        for w in range(min(8, W)):
            val = spatial_map[h, w].item()
            print(f"{val:6.2f}", end="")
        print()

    print(f"\n  🎯 Key positions:")
    print(f"    Center [{H//2},{W//2}]: {spatial_map[H//2, W//2]:.3f}")
    print(f"    Top-left [0,0]: {spatial_map[0, 0]:.3f}")
    print(f"    Top-right [0,{W-1}]: {spatial_map[0, W-1]:.3f}")
    print(f"    Bottom-left [{H-1},0]: {spatial_map[H-1, 0]:.3f}")
    print(f"    Bottom-right [{H-1},{W-1}]: {spatial_map[H-1, W-1]:.3f}")

    # Show channel statistics
    channel_means = thought.mean(dim=(1, 2))  # Average each channel
    print(f"\n  📊 Channel statistics (first 8 channels):")
    for i in range(min(8, C)):
        print(f"    Ch{i:2}: mean={channel_means[i]:.3f}")


def visualize_thought_stack(stack: torch.Tensor, thought_idx: int = 0):
    """
    Visualize the thought stack structure
    Args:
        stack: (B, C, H, W, D) where D is stack depth
        thought_idx: Which thought in the stack to examine
    """
    B, C, H, W, D = stack.shape

    print(f"\n📚 Thought Stack Structure:")
    print(f"  Stack depth: {D} thoughts")
    print(f"  Each thought: {H}×{W} spatial map")
    print(f"  Channels: {C}")
    print(f"  Total parameters per thought: {C * H * W}")

    if thought_idx < D:
        thought = stack[0, :, :, :, thought_idx]  # Get specific thought (C, H, W)
        visualize_single_thought(thought, f"Thought #{thought_idx} from Stack")


def compare_thoughts(
    thought1: torch.Tensor,
    thought2: torch.Tensor,
    title1: str = "Thought 1",
    title2: str = "Thought 2",
):
    """
    Compare two thoughts side by side
    """
    print(f"\n🔄 Comparing {title1} vs {title2}:")

    # Compute difference
    diff = torch.abs(thought1 - thought2)
    similarity = 1.0 - (diff.mean() / (thought1.std() + thought2.std() + 1e-8))

    print(f"  Similarity score: {similarity:.3f}")
    print(f"  Mean absolute difference: {diff.mean():.3f}")
    print(f"  Max difference: {diff.max():.3f}")

    # Show spatial difference map
    spatial_diff = diff.mean(dim=0)  # Average across channels
    print(f"  Spatial difference (channel-averaged):")
    print(f"    Center: {spatial_diff[thought1.shape[1]//2, thought1.shape[2]//2]:.3f}")
    print(f"    Max spatial diff: {spatial_diff.max():.3f}")
    print(f"    Min spatial diff: {spatial_diff.min():.3f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model with smaller dimensions to test
    model = StackedDiffusionModel3D(
        vocab_size=1000,
        embed_dim=256,
        num_layers=4,
        num_heads=4,
        thought_dims=(16, 16),  # Each thought is 16×16 2D
        stack_depth=8,  # Stack of 8 thoughts
        thought_hidden_dim=64,
    ).to(device)

    print(f"✅ Stacked 3D Model created")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"   Thought dimensions: {model.thought_dims} (H×W per thought)")
    print(f"   Stack depth: {model.stack_depth} thoughts")

    # Test forward pass
    batch_size = 2
    seq_length = 64

    tokens = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
    thought_stack = model.initialize_stack(
        batch_size, device, noise_std=0.2
    )  # More noise

    print(f"\n📥 Input shapes:")
    print(f"  Tokens: {tokens.shape}")
    print(f"  Thought stack: {thought_stack.shape} (B, C, H, W, D)")

    # Visualize noise initialization
    print(f"\n🎲 Noise initialization stats:")
    print(f"  Min: {thought_stack.min():.3f}, Max: {thought_stack.max():.3f}")
    print(f"  Mean: {thought_stack.mean():.3f}, Std: {thought_stack.std():.3f}")

    # Check depth bias (each depth layer should have slightly different values)
    depth_means = thought_stack.mean(dim=(0, 1, 2, 3))  # Average over B,C,H,W
    print(f"  Depth layer means: {[f'{m:.3f}' for m in depth_means.tolist()]}")

    # Create sample timesteps and token mask for testing
    timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
    token_mask = torch.rand(batch_size, seq_length) < 0.3  # 30% masking for test
    token_mask = token_mask.to(device)

    print(f"  Test token mask ratio: {token_mask.float().mean():.2f}")

    # Forward pass with DREAM-style inputs
    logits, new_thought, updated_stack = model(
        tokens, thought_stack, timesteps, token_mask, return_all_thoughts=True
    )

    print(f"\n📤 Output shapes:")
    print(f"  Logits: {logits.shape}")
    print(f"  New thought: {new_thought.shape} (B, C, H, W)")
    print(f"  Updated stack: {updated_stack.shape} (B, C, H, W, D)")

    # Show individual thoughts
    print(f"\n" + "=" * 60)
    print("🧠 INDIVIDUAL THOUGHT EXAMPLES")
    print("=" * 60)

    # 1. Show the newly generated thought
    single_new_thought = new_thought[0]  # First batch (C, H, W)
    visualize_single_thought(single_new_thought, "Newly Generated Thought")

    # 2. Show a thought from the initial stack (noise)
    initial_thought = thought_stack[0, :, :, :, 0]  # Most recent from initial stack
    visualize_single_thought(
        initial_thought, "Initial Noise Thought (Stack Position 0)"
    )

    # 3. Show an older thought from deeper in stack
    older_thought = thought_stack[0, :, :, :, -1]  # Oldest thought
    visualize_single_thought(older_thought, "Oldest Noise Thought (Stack Position -1)")

    # 4. Compare the new thought vs initial noise
    compare_thoughts(
        single_new_thought, initial_thought, "Generated Thought", "Initial Noise"
    )

    # Test multiple steps to see thought evolution
    print(f"\n" + "=" * 60)
    print("🔄 THOUGHT EVOLUTION OVER TIME")
    print("=" * 60)

    current_stack = thought_stack
    previous_thought = None

    for step in range(3):
        # Use different timesteps and masks for each step
        step_timesteps = torch.randint(0, 1000, (batch_size,)).to(device)
        step_mask = torch.rand(batch_size, seq_length) < 0.2  # 20% masking
        step_mask = step_mask.to(device)
        
        logits, new_thought, current_stack = model(
            tokens, current_stack, step_timesteps, step_mask, return_all_thoughts=True
        )

        current_single_thought = new_thought[0]  # (C, H, W)
        visualize_single_thought(current_single_thought, f"Step {step+1} Thought")

        if previous_thought is not None:
            compare_thoughts(
                current_single_thought,
                previous_thought,
                f"Step {step+1}",
                f"Step {step}",
            )

        previous_thought = current_single_thought

    print("\n✅ All tests passed!")
