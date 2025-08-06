"""
Fast-dLLM Enhanced Model with Proper Masking Support
Implements the core masking mechanism from Fast-dLLM for diffusion-based LLMs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
try:
    from .enhanced_thought_system import EnhancedThoughtStackEncoder, EnhancedThoughtStack
except ImportError:
    from enhanced_thought_system import EnhancedThoughtStackEncoder, EnhancedThoughtStack


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


class MaskedTransformerBlock(nn.Module):
    """Transformer block with masking support"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, max_seq_length=2048):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Bidirectional masked attention (DREAM-style)
        self.attn = BidirectionalMaskedAttention(embed_dim, num_heads, dropout, max_seq_length)
        self.ln1 = nn.LayerNorm(embed_dim)
        
        # Cross-attention with thought
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.ln3 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, thought_embed=None, attention_mask=None, token_mask=None):
        """
        Forward pass with masking support
        Args:
            x: input tensor (batch, seq_len, embed_dim)
            thought_embed: thought embedding (batch, embed_dim)
            attention_mask: attention mask
            token_mask: token mask (batch, seq_len) - True for masked positions
        """
        # Self-attention with masking
        attn_out, _ = self.attn(self.ln1(x), attention_mask, token_mask)
        x = x + attn_out
        
        # Cross-attention with thought
        if thought_embed is not None:
            thought_expanded = thought_embed.unsqueeze(1)  # (batch, 1, embed_dim)
            cross_out, _ = self.cross_attn(self.ln2(x), thought_expanded, thought_expanded)
            x = x + cross_out
        
        # Feed-forward
        mlp_out = self.mlp(self.ln3(x))
        x = x + mlp_out
        
        return x


class AdaptiveNoiseScheduler(nn.Module):
    """Context-adaptive token-level noise scheduling (DREAM-inspired)"""
    
    def __init__(self, embed_dim: int, num_timesteps: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_timesteps = num_timesteps
        
        # Network to predict optimal noise level per token
        self.noise_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
    def compute_token_noise_levels(self, hidden_states: torch.Tensor, 
                                  token_mask: torch.Tensor,
                                  global_timestep: int) -> torch.Tensor:
        """
        Compute adaptive noise levels for each token based on context
        Args:
            hidden_states: Token representations (batch, seq_len, embed_dim)
            token_mask: Mask indicating which tokens are masked (batch, seq_len)
            global_timestep: Global diffusion timestep
        Returns:
            Token-specific timesteps (batch, seq_len)
        """
        _, seq_len, _ = hidden_states.shape
        
        # Compute bidirectional context for each token
        if seq_len > 1:
            # Forward context: average of future tokens
            forward_context = torch.zeros_like(hidden_states)
            for i in range(seq_len - 1):
                future_states = hidden_states[:, i+1:, :]
                if future_states.size(1) > 0:
                    forward_context[:, i, :] = future_states.mean(dim=1)
                    
            # Backward context: average of past tokens  
            backward_context = torch.zeros_like(hidden_states)
            for i in range(1, seq_len):
                past_states = hidden_states[:, :i, :]
                backward_context[:, i, :] = past_states.mean(dim=1)
            
            # Combine current state with bidirectional context
            combined_features = torch.cat([
                hidden_states,
                (forward_context + backward_context) / 2
            ], dim=-1)
        else:
            # No context available, duplicate hidden states
            combined_features = torch.cat([hidden_states, hidden_states], dim=-1)
        
        # Predict noise adjustment factor for each token
        noise_factors = self.noise_predictor(combined_features).squeeze(-1)  # (batch, seq_len)
        
        # Convert to timesteps: tokens with higher factors get more noise
        # Masked tokens should generally have higher noise levels
        base_timestep = global_timestep
        
        # Adaptive timesteps: vary around the global timestep based on context
        # Range: [0.5 * global_timestep, 1.5 * global_timestep]
        adaptive_timesteps = base_timestep * (0.5 + noise_factors)
        
        # Ensure masked tokens get appropriate noise levels
        if token_mask is not None:
            # Masked tokens should have higher noise initially
            mask_boost = token_mask.float() * 0.3  # Add 30% more noise to masked tokens
            adaptive_timesteps = adaptive_timesteps * (1 + mask_boost)
        
        # Clamp to valid timestep range
        adaptive_timesteps = torch.clamp(adaptive_timesteps, 0, self.num_timesteps - 1)
        
        return adaptive_timesteps.long()


class ConfidenceMaskedDecoder(nn.Module):
    """Confidence-based masking and unmasking decoder"""
    
    def __init__(self, vocab_size, embed_dim, masking_strategy: MaskingStrategy):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.masking_strategy = masking_strategy
        
        # Confidence prediction
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Mask token embedding
        self.mask_token_embed = nn.Parameter(torch.randn(embed_dim) * 0.02)
    
    def create_initial_mask(self, seq_length: int, batch_size: int, device: torch.device, 
                           prompt_length: int = 0) -> torch.Tensor:
        """Create initial token mask - mask everything after prompt"""
        mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
        
        if prompt_length > 0:
            # Only mask tokens after the prompt
            mask[:, prompt_length:] = True
        else:
            # Mask according to mask_ratio
            num_to_mask = int(seq_length * self.masking_strategy.mask_ratio)
            mask[:, -num_to_mask:] = True
        
        return mask
    
    def compute_unmasking_confidence(self, logits: torch.Tensor, hidden_states: torch.Tensor, 
                                   token_mask: torch.Tensor) -> torch.Tensor:
        """Enhanced confidence scores with contextual information (DREAM-inspired)"""
        _, seq_len, vocab_size = logits.shape
        
        # 1. Maximum softmax probability as base confidence
        probs = F.softmax(logits, dim=-1)
        max_prob_confidence, _ = torch.max(probs, dim=-1)
        
        # 2. Entropy-based confidence (lower entropy = higher confidence)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        entropy_confidence = 1.0 - (entropy / math.log(vocab_size))  # Normalize by max entropy
        
        # 3. Learned confidence from hidden states
        learned_confidence = self.confidence_head(hidden_states).squeeze(-1)
        
        # 4. Context-aware confidence boost using bidirectional attention
        # Tokens with strong bidirectional context get confidence boost
        if seq_len > 1:
            # Compute pairwise similarities to measure context strength
            hidden_norm = F.normalize(hidden_states, dim=-1)
            context_similarity = torch.bmm(hidden_norm, hidden_norm.transpose(1, 2))
            
            # Average similarity with neighboring tokens (bidirectional context)
            context_scores = torch.zeros_like(max_prob_confidence)
            for i in range(seq_len):
                neighbors = []
                if i > 0:
                    neighbors.append(context_similarity[:, i, i-1])
                if i < seq_len - 1:
                    neighbors.append(context_similarity[:, i, i+1])
                if neighbors:
                    context_scores[:, i] = torch.stack(neighbors).mean(0)
            
            # Use context as confidence modifier
            context_boost = torch.sigmoid(context_scores * 2)  # Scale and sigmoid for [0,1] range
        else:
            context_boost = torch.ones_like(max_prob_confidence)
        
        # 5. Combine all confidence measures with DREAM-style weighting
        combined_confidence = (
            0.4 * max_prob_confidence +      # Primary: max probability
            0.2 * entropy_confidence +        # Secondary: entropy-based
            0.2 * learned_confidence +        # Learned from model
            0.2 * context_boost               # Context awareness (bidirectional)
        )
        
        # Only consider masked positions
        combined_confidence = combined_confidence * token_mask.float()
        
        return combined_confidence
    
    def update_mask_parallel(self, logits: torch.Tensor, hidden_states: torch.Tensor,
                           current_mask: torch.Tensor, step: int, total_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced block-wise parallel unmasking with DREAM-inspired improvements
        Returns: (new_mask, unmasked_tokens)
        """
        if not current_mask.any():
            return current_mask, torch.zeros_like(logits[:, :, 0], dtype=torch.long)
        
        batch_size, seq_len = current_mask.shape
        
        # Compute confidence for masked positions
        confidence = self.compute_unmasking_confidence(logits, hidden_states, current_mask)
        
        # Dynamic threshold based on generation progress
        progress_ratio = 1.0 - (step / total_steps)  # Higher at beginning
        base_threshold = self.masking_strategy.confidence_threshold
        threshold = base_threshold * (0.7 + 0.3 * progress_ratio)  # More lenient early on
        
        new_mask = current_mask.clone()
        unmasked_tokens = torch.zeros_like(logits[:, :, 0], dtype=torch.long)
        
        for batch_idx in range(batch_size):
            batch_mask = current_mask[batch_idx]
            if not batch_mask.any():
                continue
                
            batch_confidence = confidence[batch_idx]
            masked_positions = torch.where(batch_mask)[0]
            masked_confidence = batch_confidence[masked_positions]
            
            # Block-wise unmasking strategy
            block_size = self.masking_strategy.block_size
            num_masked = len(masked_positions)
            
            # Determine how many tokens to unmask this step
            if step == 0:
                # First step: unmask more aggressively
                unmask_ratio = min(0.3, 1.0 / max(1, total_steps // 2))
            else:
                # Progressive unmasking based on remaining steps
                remaining_ratio = step / total_steps
                unmask_ratio = min(0.5, remaining_ratio)
            
            num_to_unmask = max(block_size, int(num_masked * unmask_ratio))
            num_to_unmask = min(num_to_unmask, num_masked)
            
            # Strategy 1: High confidence tokens
            above_threshold = masked_confidence > threshold
            high_conf_positions = masked_positions[above_threshold]
            
            # Strategy 2: Block coherence - unmask contiguous blocks
            if len(high_conf_positions) > 0 and block_size > 1:
                # Find contiguous high-confidence regions
                positions_to_unmask = []
                for pos in high_conf_positions:
                    # Check neighboring positions for block formation
                    block_positions = []
                    for offset in range(-block_size//2, block_size//2 + 1):
                        neighbor_pos = pos + offset
                        if (neighbor_pos >= 0 and neighbor_pos < seq_len and 
                            batch_mask[neighbor_pos] and 
                            batch_confidence[neighbor_pos] > threshold * 0.8):  # Slightly lower threshold for neighbors
                            block_positions.append(neighbor_pos.item())
                    
                    positions_to_unmask.extend(block_positions)
                
                # Remove duplicates and convert to tensor
                positions_to_unmask = torch.tensor(list(set(positions_to_unmask)), device=current_mask.device)
                
                # Limit to num_to_unmask
                if len(positions_to_unmask) > num_to_unmask:
                    # Keep highest confidence ones
                    conf_values = batch_confidence[positions_to_unmask]
                    top_indices = torch.topk(conf_values, num_to_unmask).indices
                    positions_to_unmask = positions_to_unmask[top_indices]
            else:
                # Fallback: unmask top-k confident positions
                if num_to_unmask > 0:
                    top_k = min(num_to_unmask, len(masked_confidence))
                    top_indices = torch.topk(masked_confidence, top_k).indices
                    positions_to_unmask = masked_positions[top_indices]
                else:
                    # Progress guarantee: always unmask at least one
                    best_idx = torch.argmax(masked_confidence)
                    positions_to_unmask = masked_positions[best_idx:best_idx+1]
            
            # Unmask selected positions
            new_mask[batch_idx, positions_to_unmask] = False
            
            # Token sampling strategy based on step
            position_logits = logits[batch_idx, positions_to_unmask]
            
            if step < total_steps * 0.3:
                # Early: temperature sampling for diversity
                temperature = 1.0 + (0.5 * progress_ratio)  # Higher temp early
                sampled_tokens = torch.multinomial(
                    F.softmax(position_logits / temperature, dim=-1), 1
                ).squeeze(-1)
            elif step < total_steps * 0.7:
                # Middle: top-k sampling
                k = 5
                top_k_logits, top_k_indices = torch.topk(position_logits, min(k, position_logits.shape[-1]), dim=-1)
                sampled_indices = torch.multinomial(F.softmax(top_k_logits, dim=-1), 1).squeeze(-1)
                sampled_tokens = torch.gather(top_k_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)
            else:
                # Late: greedy for quality
                sampled_tokens = torch.argmax(position_logits, dim=-1)
            
            unmasked_tokens[batch_idx, positions_to_unmask] = sampled_tokens
        
        return new_mask, unmasked_tokens
    
    def apply_mask_to_embeddings(self, embeddings: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        """Apply mask to token embeddings"""
        masked_embeddings = embeddings.clone()
        # Replace masked positions with mask token embedding
        masked_embeddings[token_mask] = self.mask_token_embed
        return masked_embeddings


class ThoughtStackDecoder(nn.Module):
    """Decoder for thought stack outputs"""
    
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.decoder(x)


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


class MaskedFastDiffusionModel(nn.Module):
    """Fast-dLLM model with proper masking support"""
    
    def __init__(
        self,
        vocab_size=50257,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        max_seq_length=256,
        max_new_tokens=1024,
        thought_dims=(32, 32, 16),
        thought_stack_size=8,
        dropout=0.1,
        masking_strategy: Optional[MaskingStrategy] = None
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        self.max_new_tokens = max_new_tokens
        self.thought_dims = thought_dims
        self.thought_stack_size = thought_stack_size
        
        # Default masking strategy with Fast-dLLM settings
        if masking_strategy is None:
            masking_strategy = MaskingStrategy(
                mask_ratio=0.8,  # Start with more tokens masked
                confidence_threshold=0.7,  # More lenient than 0.9 for better unmasking
                progressive_unmasking=True
            )
        self.masking_strategy = masking_strategy
        
        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        # Learnable position embeddings that work with arbitrary generation order
        self.position_embeddings = nn.Embedding(max_seq_length, embed_dim)
        # Additional embeddings for generation order awareness
        self.order_embeddings = nn.Embedding(max_seq_length, embed_dim)  # Encodes generation order
        
        # Enhanced thought processing with self-learning
        self.thought_stack_encoder = EnhancedThoughtStackEncoder(
            thought_dims, thought_stack_size, hidden_dim=512, output_dim=embed_dim
        )
        self.thought_decoder = ThoughtStackDecoder(
            embed_dim, thought_dims, hidden_dim=512
        )
        # Use enhanced stack with importance-based retention
        self.thought_stack = EnhancedThoughtStack(thought_stack_size, thought_dims)
        
        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Masked transformer blocks
        self.transformer_blocks = nn.ModuleList([
            MaskedTransformerBlock(embed_dim, num_heads, dropout, max_seq_length)
            for _ in range(num_layers)
        ])
        
        # Output
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Masked decoder
        self.masked_decoder = ConfidenceMaskedDecoder(vocab_size, embed_dim, masking_strategy)
        
        # Adaptive noise scheduler (DREAM-inspired)
        self.adaptive_noise_scheduler = AdaptiveNoiseScheduler(embed_dim, num_timesteps=1000)
        
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
    
    def forward(self, tokens: torch.Tensor, thought_stack: torch.Tensor, 
               timestep: torch.Tensor, token_mask: Optional[torch.Tensor] = None,
               generation_order: Optional[torch.Tensor] = None,
               return_thought_losses: bool = False):
        """
        Enhanced forward pass with DREAM-inspired improvements and self-learning thoughts
        Args:
            tokens: input tokens (batch_size, seq_length)
            thought_stack: thought stack (batch_size, stack_size, *thought_dims)
            timestep: diffusion timestep (batch_size,)
            token_mask: token mask (batch_size, seq_length) - True for masked positions
            generation_order: order in which tokens were generated (batch_size, seq_length)
            return_thought_losses: whether to return self-supervised thought losses
        """
        batch_size, seq_length = tokens.shape
        device = tokens.device
        
        # Get embeddings
        token_embeds = self.token_embeddings(tokens)
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0)
        position_embeds = self.position_embeddings(position_ids)
        
        # Add generation order embeddings if available (for arbitrary order support)
        if generation_order is not None:
            order_embeds = self.order_embeddings(generation_order)
            x = token_embeds + position_embeds + order_embeds * 0.1  # Weighted order influence
        else:
            x = token_embeds + position_embeds
        
        # Apply masking to embeddings
        if token_mask is not None:
            x = self.masked_decoder.apply_mask_to_embeddings(x, token_mask)
        
        # Encode thought stack with self-supervised learning
        if return_thought_losses:
            thought_embed, thought_self_supervised = self.thought_stack_encoder(
                thought_stack, return_self_supervised=True
            )
        else:
            thought_embed = self.thought_stack_encoder(thought_stack)
        
        # Adaptive noise scheduling per token (DREAM-inspired)
        if token_mask is not None and timestep.numel() == batch_size:
            # Get initial hidden representation for context
            initial_hidden = x.clone()
            for block in self.transformer_blocks[:2]:  # Use first 2 blocks for context
                initial_hidden = block(initial_hidden, thought_embed, token_mask=token_mask)
            
            # Compute per-token noise levels
            global_timestep = timestep[0].item() if timestep.numel() > 0 else 500
            adaptive_timesteps = self.adaptive_noise_scheduler.compute_token_noise_levels(
                initial_hidden, token_mask, global_timestep
            )
            
            # Apply token-specific time embeddings
            time_embeds = []
            for b in range(batch_size):
                token_times = adaptive_timesteps[b]
                token_time_embeds = self.time_embed(token_times)
                token_time_embeds = self.time_mlp(token_time_embeds)
                time_embeds.append(token_time_embeds)
            time_embed = torch.stack(time_embeds)
            x = x + time_embed
        else:
            # Standard time embedding
            time_embed = self.time_embed(timestep)
            time_embed = self.time_mlp(time_embed)
            x = x + time_embed.unsqueeze(1)
        
        # Apply transformer blocks with bidirectional attention
        for block in self.transformer_blocks:
            x = block(x, thought_embed, token_mask=token_mask)
        
        # Output processing
        x = self.output_norm(x)
        logits = self.output_projection(x)
        
        # Generate new thought with bidirectional context
        # Use weighted pooling based on token importance
        if token_mask is not None:
            # Give more weight to unmasked tokens
            weights = (~token_mask).float()
            if weights.sum() > 0:
                weights = weights / weights.sum(dim=1, keepdim=True)
                pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
            else:
                pooled = x.mean(dim=1)
        else:
            pooled = x.mean(dim=1)
        
        new_thought = self.thought_decoder(pooled)
        
        # Return with optional thought losses for self-supervised learning
        if return_thought_losses:
            thought_losses = self.thought_stack_encoder.self_supervised.compute_losses(
                thought_self_supervised, thought_stack
            )
            return logits, new_thought, thought_losses
        
        return logits, new_thought
    
    def masked_generate(self, initial_stack=None, prompt_tokens=None, num_steps=50, 
                       temperature=1.0) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Generation with proper Fast-dLLM masking strategy
        """
        batch_size = 1 if initial_stack is None else initial_stack.shape[0]
        device = next(self.parameters()).device
        
        # Initialize thought stack
        if initial_stack is None:
            thought_stack = self.thought_stack.initialize_stack(batch_size)
            self.thought_stack.device = device
            thought_stack = thought_stack.to(device)
        else:
            thought_stack = initial_stack
        
        # Setup sequence
        if prompt_tokens is None:
            seq_length = self.max_seq_length
            tokens = torch.randint(0, self.vocab_size, (batch_size, seq_length), device=device)
            prompt_length = 0
        else:
            prompt_length = prompt_tokens.shape[1]
            output_length = min(self.max_new_tokens, self.max_seq_length - prompt_length)
            
            # Initialize output tokens randomly
            output_tokens = torch.randint(0, self.vocab_size, (batch_size, output_length), device=device)
            tokens = torch.cat([prompt_tokens, output_tokens], dim=1)
            seq_length = tokens.shape[1]
        
        # Create initial mask
        token_mask = self.masked_decoder.create_initial_mask(
            seq_length, batch_size, device, prompt_length
        )
        
        # Track statistics
        stats = {
            'total_unmasked': 0,
            'steps_taken': 0,
            'final_mask_ratio': 0.0
        }
        
        self.eval()
        with torch.no_grad():
            # Masked diffusion process
            for step in reversed(range(num_steps)):
                timestep = torch.tensor([step], device=device).repeat(batch_size)
                
                # Forward pass with current mask
                logits, new_thought = self.forward(tokens, thought_stack, timestep, token_mask)
                
                # Update thought stack with importance weighting if available
                if hasattr(self.thought_stack, 'push_thought_with_importance'):
                    # Use enhanced stack with importance scores
                    thought_stack = self.thought_stack.push_thought_with_importance(
                        thought_stack, new_thought
                    )
                else:
                    # Fallback to standard push
                    thought_stack = self.thought_stack.push_thought(thought_stack, new_thought)
                
                # Update mask and tokens using confidence-based parallel unmasking
                if step > 0:
                    # Get hidden states for confidence computation
                    hidden_states = self.transformer_blocks[-1](
                        self.token_embeddings(tokens) + 
                        self.position_embeddings(torch.arange(seq_length, device=device).unsqueeze(0)),
                        self.thought_stack_encoder(thought_stack),
                        token_mask=token_mask
                    )
                    
                    new_mask, unmasked_tokens = self.masked_decoder.update_mask_parallel(
                        logits, hidden_states, token_mask, num_steps - step, num_steps
                    )
                    
                    # Update tokens at unmasked positions
                    update_positions = (token_mask & ~new_mask)
                    tokens[update_positions] = unmasked_tokens[update_positions]
                    
                    # Update mask
                    token_mask = new_mask
                    
                    # Update stats
                    stats['total_unmasked'] += update_positions.sum().item()
                else:
                    # Final step - unmask everything and sample
                    remaining_mask = token_mask.clone()
                    if remaining_mask.any():
                        final_logits = logits[remaining_mask]
                        final_tokens = torch.multinomial(F.softmax(final_logits / temperature, dim=-1), 1).squeeze(-1)
                        tokens[remaining_mask] = final_tokens
                    
                    token_mask.fill_(False)  # Everything unmasked
                
                stats['steps_taken'] = num_steps - step
        
        stats['final_mask_ratio'] = token_mask.float().mean().item()
        
        return tokens, thought_stack, stats
    
    def initialize_thought_stack(self, batch_size: int, device: torch.device):
        """Initialize thought stack"""
        self.thought_stack.device = device
        return self.thought_stack.initialize_stack(batch_size)


if __name__ == "__main__":
    # Test masked model
    masking_strategy = MaskingStrategy(
        mask_ratio=0.7,
        confidence_threshold=0.75,
        progressive_unmasking=True
    )
    
    model = MaskedFastDiffusionModel(
        vocab_size=1000,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        max_seq_length=64,
        thought_dims=(8, 8, 4),
        masking_strategy=masking_strategy
    )
    
    print(f"✅ Masked Fast-dLLM model created with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
    # Test masked generation
    prompt = torch.randint(0, 1000, (1, 16))
    print(f"Prompt shape: {prompt.shape}")
    
    tokens, final_stack, stats = model.masked_generate(
        prompt_tokens=prompt,
        num_steps=25,
        temperature=0.8
    )
    
    print(f"Generated tokens shape: {tokens.shape}")
    print(f"Stats: {stats}")
    print("✅ Masked generation test passed!")