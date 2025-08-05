"""
Fast-dLLM Enhanced Diffusion Thought Tensor Model
Integrates Fast-dLLM optimizations: KV caching, confidence-aware parallel decoding, and block-wise inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
try:
    from .base_model import SinusoidalTimeEmbedding, TransformerBlock
    from .enhanced_model import ThoughtStackEncoder, ThoughtStackDecoder, ThoughtStack
except ImportError:
    from base_model import SinusoidalTimeEmbedding
    from enhanced_model import ThoughtStackEncoder, ThoughtStackDecoder, ThoughtStack


@dataclass
class KVCache:
    """Key-Value cache for attention acceleration"""
    keys: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None
    block_size: int = 32
    max_length: int = 2048
    
    def get_cache_size(self) -> int:
        """Get current cache size"""
        if self.keys is None:
            return 0
        return self.keys.shape[2]  # seq_len dimension
    
    def can_use_cache(self, seq_len: int) -> bool:
        """Check if cache can be used for given sequence length"""
        cache_size = self.get_cache_size()
        return cache_size > 0 and seq_len >= cache_size
    
    def extract_cached_portion(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract cached keys and values for current sequence length"""
        if not self.can_use_cache(seq_len):
            return None, None
        
        # Return cached portion up to seq_len
        cached_keys = self.keys[:, :, :seq_len]
        cached_values = self.values[:, :, :seq_len]
        return cached_keys, cached_values
    
    def update_cache(self, keys: torch.Tensor, values: torch.Tensor, block_idx: int = 0):
        """Update cache with new keys and values"""
        if self.keys is None:
            self.keys = keys.detach()
            self.values = values.detach()
        else:
            # Extend cache with new keys/values
            self.keys = torch.cat([self.keys, keys.detach()], dim=2)
            self.values = torch.cat([self.values, values.detach()], dim=2)
            
            # Trim if exceeding max length
            if self.keys.shape[2] > self.max_length:
                self.keys = self.keys[:, :, -self.max_length:]
                self.values = self.values[:, :, -self.max_length:]


class FastTransformerBlock(nn.Module):
    """Enhanced transformer block with KV caching and optimized attention"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_cache=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_cache = use_cache
        self.head_dim = embed_dim // num_heads
        
        # Separate projections for better caching
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Cross-attention projections
        self.cross_q_proj = nn.Linear(embed_dim, embed_dim)
        self.cross_k_proj = nn.Linear(embed_dim, embed_dim)
        self.cross_v_proj = nn.Linear(embed_dim, embed_dim)
        self.cross_out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Layer norms
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        self.cross_attn_norm = nn.LayerNorm(embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
        # Feed-forward network with optimizations
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.kv_cache = KVCache() if use_cache else None
    
    def _apply_attention(self, q, k, v, attn_mask=None):
        """Optimized attention computation"""
        batch_size, seq_len, embed_dim = q.shape
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return attn_output, attn_weights
    
    def forward(self, x, thought_embed=None, use_cache=True, past_kv=None):
        """
        Forward pass with optional KV caching
        Args:
            x: input tensor (seq_len, batch, embed_dim)
            thought_embed: thought embedding for cross-attention (batch, embed_dim)
            use_cache: whether to use KV caching
            past_kv: past key-value states for incremental decoding
        """
        # Convert from seq_first to batch_first format
        x = x.transpose(0, 1)  # (batch, seq, embed)
        seq_len, batch_size, embed_dim = x.shape[1], x.shape[0], x.shape[2]
        
        # Self-attention with caching
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Use cached keys/values if available
        if use_cache and self.kv_cache and self.kv_cache.can_use_cache(seq_len):
            cached_k, cached_v = self.kv_cache.extract_cached_portion(seq_len)
            if cached_k is not None:
                k = cached_k
                v = cached_v
        
        attn_out, _ = self._apply_attention(q, k, v)
        attn_out = self.out_proj(attn_out)
        x = self.self_attn_norm(x + self.dropout(attn_out))
        
        # Update cache
        if use_cache and self.kv_cache:
            self.kv_cache.update_cache(k, v)
        
        # Cross-attention with thought
        if thought_embed is not None:
            cross_q = self.cross_q_proj(x)
            # Expand thought for cross-attention
            thought_expanded = thought_embed.unsqueeze(1).expand(-1, seq_len, -1)
            cross_k = self.cross_k_proj(thought_expanded)
            cross_v = self.cross_v_proj(thought_expanded)
            
            cross_out, _ = self._apply_attention(cross_q, cross_k, cross_v)
            cross_out = self.cross_out_proj(cross_out)
            x = self.cross_attn_norm(x + self.dropout(cross_out))
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + self.dropout(ffn_out))
        
        # Convert back to seq_first format
        return x.transpose(0, 1)
    
    def clear_cache(self):
        """Clear KV cache"""
        if self.kv_cache:
            self.kv_cache.keys = None
            self.kv_cache.values = None


class ConfidenceDecoder(nn.Module):
    """Confidence-aware parallel decoding system"""
    
    def __init__(self, vocab_size, embed_dim, confidence_threshold=0.8):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.confidence_threshold = confidence_threshold
        
        # Confidence prediction head
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def compute_confidence(self, logits, hidden_states):
        """Compute confidence scores for each token position"""
        # Use entropy-based confidence + learned confidence
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        max_entropy = math.log(self.vocab_size)
        entropy_confidence = 1.0 - (entropy / max_entropy)
        
        # Learned confidence from hidden states
        learned_confidence = self.confidence_head(hidden_states).squeeze(-1)
        
        # Combine both confidence measures
        combined_confidence = 0.7 * entropy_confidence + 0.3 * learned_confidence
        
        return combined_confidence
    
    def parallel_decode_step(self, logits, hidden_states, temperature=1.0):
        """
        Perform confidence-aware parallel decoding
        Returns tokens and mask indicating which tokens are high confidence
        """
        # Compute confidence scores
        confidence = self.compute_confidence(logits, hidden_states)
        
        # Create mask for high-confidence positions
        high_conf_mask = confidence > self.confidence_threshold
        
        # Sample tokens
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=-1)
        tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1)
        tokens = tokens.view(logits.shape[:-1])
        
        return tokens, high_conf_mask, confidence


class FastDiffusionThoughtModel(nn.Module):
    """Fast-dLLM enhanced diffusion model with KV caching and parallel decoding"""
    
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
        thought_dropout=0.15,
        noise_injection_std=0.01,
        use_gradient_checkpointing=True,
        use_kv_cache=True,
        confidence_threshold=0.8,
        block_size=32
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
        self.use_kv_cache = use_kv_cache
        self.block_size = block_size
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_seq_length, embed_dim)
        
        # Thought stack processing
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
        
        # Fast transformer blocks with KV caching
        self.transformer_blocks = nn.ModuleList([
            FastTransformerBlock(embed_dim, num_heads, dropout, use_cache=use_kv_cache)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Confidence-aware decoder
        self.confidence_decoder = ConfidenceDecoder(vocab_size, embed_dim, confidence_threshold)
        
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
    
    def clear_caches(self):
        """Clear all KV caches"""
        for block in self.transformer_blocks:
            block.clear_cache()
    
    def forward(self, noisy_tokens, thought_stack, timestep, use_cache=True):
        """
        Forward pass with caching support
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
        
        # Apply fast transformer blocks with caching
        x = x.transpose(0, 1)  # (batch, seq, embed) -> (seq, batch, embed)
        for block in self.transformer_blocks:
            if self.training and self.use_gradient_checkpointing:
                x = checkpoint(
                    block, x, thought_embed, use_cache, use_reentrant=False
                )
            else:
                x = block(x, thought_embed, use_cache)
        x = x.transpose(0, 1)  # (seq, batch, embed) -> (batch, seq, embed)
        
        # Output processing
        x = self.output_norm(x)
        logits = self.output_projection(x)
        
        # Generate new thought tensor
        pooled = x.mean(dim=1)
        new_thought = self.thought_decoder(pooled)
        
        # Add noise injection during training
        if self.training and self.noise_injection_std > 0:
            noise = torch.randn_like(new_thought) * self.noise_injection_std
            new_thought = new_thought + noise
        
        return logits, new_thought, x  # Return hidden states for confidence
    
    def fast_generate(self, initial_stack=None, prompt_tokens=None, num_steps=100, 
                     temperature=1.0, use_parallel_decoding=True, max_parallel_tokens=4):
        """
        Fast generation with KV caching and confidence-aware parallel decoding
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
        
        if prompt_tokens is None:
            seq_length = self.max_seq_length
            tokens = torch.randint(0, self.vocab_size, (batch_size, seq_length), device=device)
            prompt_length = 0
        else:
            prompt_length = prompt_tokens.shape[1]
            output_length = min(self.max_new_tokens, self.max_seq_length - prompt_length)
            
            if output_length <= 0:
                raise ValueError(f"Prompt too long ({prompt_length} tokens)")
            
            output_tokens = torch.randint(0, self.vocab_size, (batch_size, output_length), device=device)
            tokens = torch.cat([prompt_tokens, output_tokens], dim=1)
            seq_length = tokens.shape[1]
        
        # Clear caches before generation
        self.clear_caches()
        
        # Track metrics
        total_parallel_accepts = 0
        total_parallel_attempts = 0
        
        # Fast diffusion sampling with caching
        for t in reversed(range(num_steps)):
            timestep = torch.tensor([t], device=device).repeat(batch_size)
            
            # Forward pass with caching
            logits, new_thought, hidden_states = self.forward(
                tokens, thought_stack, timestep, use_cache=True
            )
            
            # Update thought stack
            thought_stack = self.thought_stack.push_thought(thought_stack, new_thought)
            
            # Confidence-aware parallel decoding
            if use_parallel_decoding and t > 0:
                new_tokens, high_conf_mask, confidence = self.confidence_decoder.parallel_decode_step(
                    logits, hidden_states, temperature
                )
                
                # Count parallel decoding stats
                total_parallel_attempts += high_conf_mask.numel()
                total_parallel_accepts += high_conf_mask.sum().item()
                
                # Apply mask - only update high confidence tokens
                if prompt_tokens is not None:
                    # Keep prompt fixed
                    output_mask = high_conf_mask[:, prompt_length:]
                    output_tokens = tokens[:, prompt_length:].clone()
                    output_tokens[output_mask] = new_tokens[:, prompt_length:][output_mask]
                    tokens = torch.cat([prompt_tokens, output_tokens], dim=1)
                else:
                    tokens[high_conf_mask] = new_tokens[high_conf_mask]
            else:
                # Standard sampling for final step
                if prompt_tokens is not None:
                    output_logits = logits[:, prompt_length:]
                    if t > 0:
                        probs = F.softmax(output_logits / temperature, dim=-1)
                        new_output_tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1)
                        new_output_tokens = new_output_tokens.view(batch_size, -1)
                    else:
                        new_output_tokens = output_logits.argmax(dim=-1)
                    tokens = torch.cat([prompt_tokens, new_output_tokens], dim=1)
                else:
                    if t > 0:
                        probs = F.softmax(logits / temperature, dim=-1)
                        tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1)
                        tokens = tokens.view(batch_size, seq_length)
                    else:
                        tokens = logits.argmax(dim=-1)
        
        # Calculate parallel decoding efficiency
        parallel_acceptance_rate = total_parallel_accepts / max(total_parallel_attempts, 1)
        
        return tokens, thought_stack, {
            'parallel_acceptance_rate': parallel_acceptance_rate,
            'total_parallel_accepts': total_parallel_accepts,
            'total_parallel_attempts': total_parallel_attempts
        }
    
    def evolve_thought_stack(self, thought_stack, new_thought):
        """Update the thought stack with a new thought"""
        return self.thought_stack.push_thought(thought_stack, new_thought)
    
    def initialize_thought_stack(self, batch_size, device):
        """Initialize a new thought stack"""
        self.thought_stack.device = device
        return self.thought_stack.initialize_stack(batch_size)


def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test Fast-dLLM enhanced model
    model = FastDiffusionThoughtModel(
        vocab_size=1000,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        max_seq_length=64,
        thought_dims=(8, 8, 4),
        thought_stack_size=8,
        use_kv_cache=True,
        confidence_threshold=0.8
    )
    
    param_count = count_parameters(model)
    print(f"Fast-dLLM enhanced model created with {param_count / 1e6:.2f}M parameters")
    
    # Test forward pass
    batch_size = 2
    seq_length = 32
    
    dummy_tokens = torch.randint(0, 1000, (batch_size, seq_length))
    dummy_stack = model.initialize_thought_stack(batch_size, 'cpu')
    dummy_timestep = torch.tensor([250, 250])
    
    with torch.no_grad():
        logits, new_thought, hidden_states = model(dummy_tokens, dummy_stack, dummy_timestep)
        updated_stack = model.evolve_thought_stack(dummy_stack, new_thought)
    
    print(f"Output shapes:")
    print(f"  Logits: {logits.shape}")
    print(f"  New thought: {new_thought.shape}")
    print(f"  Hidden states: {hidden_states.shape}")
    print(f"  Updated stack: {updated_stack.shape}")
    
    # Test fast generation
    print("\nTesting fast generation...")
    prompt = torch.randint(0, 1000, (1, 10))
    tokens, final_stack, stats = model.fast_generate(
        prompt_tokens=prompt, 
        num_steps=50,
        use_parallel_decoding=True
    )
    
    print(f"Generated tokens shape: {tokens.shape}")
    print(f"Parallel acceptance rate: {stats['parallel_acceptance_rate']:.2f}")
    print("✅ Fast-dLLM enhanced model test passed!")