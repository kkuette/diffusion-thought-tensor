"""
Fast-dLLM Enhanced Model with Proper Masking Support
Implements the core masking mechanism from Fast-dLLM for diffusion-based LLMs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
try:
    from .base_model import SinusoidalTimeEmbedding
    from .enhanced_model import ThoughtStackEncoder, ThoughtStackDecoder, ThoughtStack
except ImportError:
    from base_model import SinusoidalTimeEmbedding
    from enhanced_model import ThoughtStackEncoder, ThoughtStackDecoder, ThoughtStack


@dataclass
class MaskingStrategy:
    """Configuration for token masking strategies"""
    mask_ratio: float = 0.5  # Ratio of tokens to mask initially
    confidence_threshold: float = 0.8  # Threshold for unmasking tokens
    progressive_unmasking: bool = True  # Whether to progressively unmask
    block_size: int = 1  # Size of blocks to unmask together
    mask_token_id: int = -1  # Special token ID for masked positions


class CausalMaskedAttention(nn.Module):
    """Causal attention with support for token masking"""
    
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
        
        # Causal mask - register as buffer so it moves with model
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_length, max_seq_length)).view(
                1, 1, max_seq_length, max_seq_length
            )
        )
    
    def forward(self, x, attention_mask=None, token_mask=None):
        """
        Forward pass with causal and token masking
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
        
        # Apply causal mask
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Expand to match scores dimensions
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply token masking - prevent masked tokens from attending to future positions
        # but allow them to attend to past positions and receive thought information
        if token_mask is not None:
            # Only mask future positions for masked tokens (preserve causal structure)
            token_mask_expanded = token_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            future_mask = torch.triu(torch.ones_like(scores[0, 0]), diagonal=1).bool()
            combined_mask = token_mask_expanded & future_mask.unsqueeze(0)
            scores = scores.masked_fill(combined_mask, float('-inf'))
        
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
        
        # Masked attention
        self.attn = CausalMaskedAttention(embed_dim, num_heads, dropout, max_seq_length)
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
        """Compute confidence scores for unmasking decisions using Fast-dLLM approach"""
        # Fast-dLLM uses maximum softmax probability as confidence
        probs = F.softmax(logits, dim=-1)
        max_prob_confidence, _ = torch.max(probs, dim=-1)
        
        # Optional: Also use learned confidence from hidden states
        learned_confidence = self.confidence_head(hidden_states).squeeze(-1)
        
        # Combine confidences (weighted toward max probability as in Fast-dLLM)
        combined_confidence = 0.8 * max_prob_confidence + 0.2 * learned_confidence
        
        # Only consider masked positions
        combined_confidence = combined_confidence * token_mask.float()
        
        return combined_confidence
    
    def update_mask_parallel(self, logits: torch.Tensor, hidden_states: torch.Tensor,
                           current_mask: torch.Tensor, step: int, total_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update mask using Fast-dLLM parallel confidence-based unmasking
        Returns: (new_mask, unmasked_tokens)
        """
        if not current_mask.any():
            return current_mask, torch.zeros_like(logits[:, :, 0], dtype=torch.long)
        
        # Compute confidence for masked positions
        confidence = self.compute_unmasking_confidence(logits, hidden_states, current_mask)
        
        # Use fixed threshold as in Fast-dLLM (0.9 default, but make it more lenient for training)
        threshold = self.masking_strategy.confidence_threshold
        
        new_mask = current_mask.clone()
        unmasked_tokens = torch.zeros_like(logits[:, :, 0], dtype=torch.long)
        
        for batch_idx in range(logits.shape[0]):
            batch_mask = current_mask[batch_idx]
            if not batch_mask.any():
                continue
                
            batch_confidence = confidence[batch_idx]
            masked_positions = torch.where(batch_mask)[0]
            masked_confidence = batch_confidence[masked_positions]
            
            # Find tokens above threshold
            above_threshold = masked_confidence > threshold
            
            if above_threshold.any():
                # Unmask all tokens above threshold (Fast-dLLM approach)
                positions_to_unmask = masked_positions[above_threshold]
            else:
                # Progress guarantee: always unmask at least the most confident token
                best_idx = torch.argmax(masked_confidence)
                positions_to_unmask = masked_positions[best_idx:best_idx+1]
            
            # Unmask these positions
            new_mask[batch_idx, positions_to_unmask] = False
            
            # Sample tokens for unmasked positions using greedy or sampling
            position_logits = logits[batch_idx, positions_to_unmask]
            if step < total_steps // 2:
                # Early steps: sample for diversity
                sampled_tokens = torch.multinomial(F.softmax(position_logits, dim=-1), 1).squeeze(-1)
            else:
                # Later steps: greedy for quality
                sampled_tokens = torch.argmax(position_logits, dim=-1)
            unmasked_tokens[batch_idx, positions_to_unmask] = sampled_tokens
        
        return new_mask, unmasked_tokens
    
    def apply_mask_to_embeddings(self, embeddings: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        """Apply mask to token embeddings"""
        masked_embeddings = embeddings.clone()
        # Replace masked positions with mask token embedding
        masked_embeddings[token_mask] = self.mask_token_embed
        return masked_embeddings


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
        self.position_embeddings = nn.Embedding(max_seq_length, embed_dim)
        
        # Thought processing
        self.thought_stack_encoder = ThoughtStackEncoder(
            thought_dims, thought_stack_size, hidden_dim=512, output_dim=embed_dim
        )
        self.thought_decoder = ThoughtStackDecoder(
            embed_dim, thought_dims, hidden_dim=512
        )
        self.thought_stack = ThoughtStack(thought_stack_size, thought_dims)
        
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
               timestep: torch.Tensor, token_mask: Optional[torch.Tensor] = None):
        """
        Forward pass with masking support
        Args:
            tokens: input tokens (batch_size, seq_length)
            thought_stack: thought stack (batch_size, stack_size, *thought_dims)
            timestep: diffusion timestep (batch_size,)
            token_mask: token mask (batch_size, seq_length) - True for masked positions
        """
        batch_size, seq_length = tokens.shape
        device = tokens.device
        
        # Get embeddings
        token_embeds = self.token_embeddings(tokens)
        position_ids = torch.arange(seq_length, device=device).unsqueeze(0)
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine embeddings
        x = token_embeds + position_embeds
        
        # Apply masking to embeddings
        if token_mask is not None:
            x = self.masked_decoder.apply_mask_to_embeddings(x, token_mask)
        
        # Encode thought stack
        thought_embed = self.thought_stack_encoder(thought_stack)
        
        # Add time embedding
        time_embed = self.time_embed(timestep)
        time_embed = self.time_mlp(time_embed)
        x = x + time_embed.unsqueeze(1)
        
        # Apply masked transformer blocks
        for block in self.transformer_blocks:
            x = block(x, thought_embed, token_mask=token_mask)
        
        # Output processing
        x = self.output_norm(x)
        logits = self.output_projection(x)
        
        # Generate new thought
        pooled = x.mean(dim=1)
        new_thought = self.thought_decoder(pooled)
        
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
                
                # Update thought stack
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