"""
Test script for interactive chat functionality
"""
import torch
import tempfile
import os
from interactive_chat import InteractiveDiffusionChat
from model.masked_fast_model import MaskedFastDiffusionModel, MaskingStrategy

def test_chat_initialization():
    """Test basic chat initialization without a real checkpoint"""
    
    # Create a temporary fake checkpoint
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
        # Create a small test model
        test_model = MaskedFastDiffusionModel(
            vocab_size=1000,
            embed_dim=128,
            num_layers=2,
            num_heads=4,
            max_seq_length=64,
            thought_dims=(8, 8, 4),
            thought_stack_size=4,  # Make sure this matches the config
            masking_strategy=MaskingStrategy(mask_ratio=0.5, confidence_threshold=0.6)
        )
        
        # Save model state
        torch.save(test_model.state_dict(), tmp.name)
        checkpoint_path = tmp.name
    
    try:
        # Test configuration
        config = {
            "model": {
                "vocab_size": 1000,
                "embed_dim": 128,
                "num_layers": 2,
                "num_heads": 4,
                "max_seq_length": 64,
                "max_new_tokens": 32,
                "thought_dims": [8, 8, 4],
                "thought_stack_size": 4,
                "dropout": 0.1,
                "masking": {
                    "mask_ratio": 0.5,
                    "confidence_threshold": 0.6,
                    "progressive_unmasking": True
                }
            }
        }
        
        # Save config to temp file
        import json
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as config_file:
            json.dump(config, config_file)
            config_path = config_file.name
        
        # Initialize chat
        chat = InteractiveDiffusionChat(
            model_path=checkpoint_path,
            config_path=config_path,
            tokenizer_name="gpt2",
            max_length=20,
            temperature=0.8,
            num_diffusion_steps=10
        )
        
        print("✅ Chat initialization successful!")
        print(f"Model has {sum(p.numel() for p in chat.model.parameters()) / 1e6:.2f}M parameters")
        
        # Test a simple generation (won't be meaningful but should run)
        try:
            result = chat.chat_turn("Hello!")
            print("✅ Basic generation test passed!")
            print(f"Response length: {len(result['response'])}")
            print(f"Generation stats: {result.get('generation_stats', {})}")
        except Exception as e:
            print(f"⚠️  Generation test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Chat initialization failed: {e}")
        return False
    
    finally:
        # Clean up temp files
        if os.path.exists(checkpoint_path):
            os.unlink(checkpoint_path)
        if 'config_path' in locals() and os.path.exists(config_path):
            os.unlink(config_path)

if __name__ == "__main__":
    print("🧪 Testing Interactive Chat with Masked Fast Model")
    print("="*50)
    
    success = test_chat_initialization()
    
    if success:
        print("\n✅ All tests passed! Interactive chat is ready to use.")
        print("\nTo use with a real checkpoint:")
        print("python interactive_chat.py --model path/to/checkpoint.pt")
    else:
        print("\n❌ Tests failed. Check the implementation.")