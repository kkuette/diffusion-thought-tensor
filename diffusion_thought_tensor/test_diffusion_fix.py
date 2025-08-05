#!/usr/bin/env python3
"""
Test the fixed diffusion generation with separate input/output windows
"""

import torch
import yaml
from model.enhanced_model import EnhancedDiffusionThoughtModel

def test_diffusion_generation():
    """Test the new diffusion generation behavior"""
    
    # Load config
    with open('configs/enhanced_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = EnhancedDiffusionThoughtModel(
        vocab_size=config["model"]["vocab_size"],
        embed_dim=config["model"]["embed_dim"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        max_seq_length=config["model"]["max_seq_length"],
        max_new_tokens=config["model"]["max_new_tokens"],
        thought_dims=tuple(config["thought_tensor"]["input_dims"]),
        thought_stack_size=config["thought_tensor"]["stack_size"],
        dropout=config["model"]["dropout"],
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    print(f"Max sequence length: {model.max_seq_length}")
    print(f"Max new tokens: {model.max_new_tokens}")
    
    # Test 1: Generation without prompt (full sequence)
    print("\n=== Test 1: Full sequence generation ===")
    tokens, thought_stack, thought_history = model.generate_with_stack(
        prompt_tokens=None,
        num_steps=10,  # Short for testing
        temperature=1.0
    )
    print(f"Generated sequence shape: {tokens.shape}")
    print(f"Expected: (1, {model.max_seq_length})")
    assert tokens.shape == (1, model.max_seq_length), "Full sequence generation failed"
    
    # Test 2: Generation with prompt (prompt + output)
    print("\n=== Test 2: Prompt + output generation ===")
    prompt_length = 50
    prompt_tokens = torch.randint(0, model.vocab_size, (1, prompt_length), device=device)
    
    tokens, thought_stack, thought_history = model.generate_with_stack(
        prompt_tokens=prompt_tokens,
        num_steps=10,
        temperature=1.0
    )
    
    expected_length = prompt_length + model.max_new_tokens
    print(f"Prompt length: {prompt_length}")
    print(f"Generated sequence shape: {tokens.shape}")
    print(f"Expected: (1, {expected_length})")
    
    # Check that prompt is preserved
    prompt_preserved = torch.equal(tokens[:, :prompt_length], prompt_tokens)
    print(f"Prompt preserved: {prompt_preserved}")
    
    assert tokens.shape == (1, expected_length), "Prompt + output generation failed"
    assert prompt_preserved, "Prompt was not preserved during generation"
    
    # Test 3: Long prompt (should handle gracefully)
    print("\n=== Test 3: Long prompt handling ===")
    long_prompt_length = model.max_seq_length - 10  # Very long prompt
    long_prompt_tokens = torch.randint(0, model.vocab_size, (1, long_prompt_length), device=device)
    
    try:
        tokens, _, _ = model.generate_with_stack(
            prompt_tokens=long_prompt_tokens,
            num_steps=5,
            temperature=1.0
        )
        expected_output_length = min(model.max_new_tokens, model.max_seq_length - long_prompt_length)
        expected_total_length = long_prompt_length + expected_output_length
        
        print(f"Long prompt length: {long_prompt_length}")
        print(f"Generated sequence shape: {tokens.shape}")
        print(f"Expected: (1, {expected_total_length})")
        
        assert tokens.shape == (1, expected_total_length), "Long prompt handling failed"
        
        # Check prompt preservation
        prompt_preserved = torch.equal(tokens[:, :long_prompt_length], long_prompt_tokens)
        print(f"Long prompt preserved: {prompt_preserved}")
        assert prompt_preserved, "Long prompt was not preserved"
        
    except ValueError as e:
        print(f"Handled long prompt gracefully: {e}")
    
    print("\n✅ All tests passed! Diffusion generation fix working correctly.")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Input window (max_seq_length): {model.max_seq_length} tokens")
    print(f"Output window (max_new_tokens): {model.max_new_tokens} tokens")
    print(f"Total capacity: {model.max_seq_length} tokens")
    print(f"Generation behavior: Fixed prompt + denoised output")

if __name__ == "__main__":
    test_diffusion_generation()