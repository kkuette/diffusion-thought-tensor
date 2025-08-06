"""
Interactive Multi-Turn Chat with DREAM-Enhanced Diffusion Thought Tensor Model
Load a trained DREAM checkpoint and conduct conversations with 3D thought evolution tracking
"""

import torch
import os
import sys
import json
import yaml
from typing import Dict, Optional, Tuple
from datetime import datetime
import argparse

# Import model and utilities
from model.stacked_3d_model import StackedDiffusionModel3D, MaskingStrategy

try:
    from transformers import AutoTokenizer

    HF_AVAILABLE = True
except ImportError:
    print(
        "Warning: Hugging Face transformers not available. Install with: pip install transformers"
    )
    HF_AVAILABLE = False


class InteractiveDiffusionChat:
    """Interactive chat interface with diffusion thought tensor model"""

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        tokenizer_name: str = "gpt2",
        max_length: int = 100,
        temperature: float = 0.8,
        num_diffusion_steps: int = 50,
        use_fixed_timestep: bool = True,
    ):
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.temperature = temperature
        self.num_diffusion_steps = num_diffusion_steps
        self.use_fixed_timestep = use_fixed_timestep
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load configuration
        if config_path and os.path.exists(config_path):
            self.config = self._load_config(config_path)
        else:
            self.config = self._default_config()

        # Initialize tokenizer
        if not HF_AVAILABLE:
            raise ImportError("Hugging Face transformers required for tokenization")

        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        print(f"Loading model from: {model_path}")
        self.model = self._load_model()
        self.model.eval()

        # Initialize conversation state
        self.conversation_history = []
        self.thought_stack = None
        self.thought_history = []

        print(f"✅ Interactive chat ready on {self.device}")
        print(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M"
        )

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, "r") as f:
            if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                return yaml.safe_load(f)
            else:
                return json.load(f)

    def _default_config(self) -> Dict:
        """Default configuration if no config file provided"""
        return {
            "model": {
                "vocab_size": 50257,
                "embed_dim": 256,  # Match DREAM config
                "num_layers": 6,  # Match DREAM config
                "num_heads": 8,  # Match DREAM config
                "max_seq_length": 512,  # Match DREAM config
                "max_new_tokens": 256,
                "dropout": 0.1,  # Match DREAM config
            },
            "thought_tensor": {
                "input_dims": [16, 16, 128],  # DREAM 3D format: [H, W, channels]
                "stack_size": 8,  # Depth of 3D thought stack
                "hidden_dim": 256,  # Match embed_dim
                "dropout": 0.15,
            },
            "masking": {
                "mask_ratio": 0.7,
                "confidence_threshold": 0.75,
                "progressive_unmasking": True,
                "block_size": 4,
                "adaptive_masking": True,
            },
            "dream": {
                "bidirectional_attention": True,
                "pad_token_id": 50256,
            },
        }

    def _load_model(self) -> StackedDiffusionModel3D:
        """Load model from checkpoint"""

        # Create model with configuration
        model_config = self.config["model"]
        thought_config = self.config["thought_tensor"]
        masking_config = self.config.get("masking", {})

        # Create masking strategy
        masking_strategy = MaskingStrategy(
            mask_ratio=masking_config.get("mask_ratio", 0.7),
            confidence_threshold=masking_config.get("confidence_threshold", 0.75),
            progressive_unmasking=masking_config.get("progressive_unmasking", True),
            block_size=masking_config.get("block_size", 4),
        )

        # Map 3D config to 2D spatial dimensions (H, W) and separate depth
        original_dims = thought_config["input_dims"]  # e.g., [16, 16, 128]
        thought_dims = (original_dims[0], original_dims[1])  # (H, W)
        stack_depth = thought_config["stack_size"]
        thought_hidden_dim = original_dims[2] if len(original_dims) > 2 else 256

        model = StackedDiffusionModel3D(
            vocab_size=model_config["vocab_size"],
            embed_dim=model_config["embed_dim"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            thought_dims=thought_dims,  # (H, W) for each 2D thought
            stack_depth=stack_depth,  # Number of thoughts in the 3D stack
            thought_hidden_dim=thought_hidden_dim,  # Channel dimension for thoughts
            max_seq_length=model_config["max_seq_length"],
            dropout=model_config["dropout"],
            masking_strategy=masking_strategy,
        )

        # Load checkpoint
        if os.path.exists(self.model_path):
            # Use weights_only=False for compatibility with older checkpoints
            checkpoint = torch.load(
                self.model_path, map_location=self.device, weights_only=False
            )

            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                if "epoch" in checkpoint:
                    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
            else:
                # Assume the checkpoint is just the state dict
                model.load_state_dict(checkpoint)

            print(f"✅ Model loaded from {self.model_path}")
            # Verify the model is actually trained by checking weight statistics
            first_layer_weight = list(model.parameters())[0]
            weight_std = first_layer_weight.std().item()
            print(f"   First layer weight std: {weight_std:.4f}")
            if weight_std < 0.01:
                print("   ⚠️  WARNING: Model weights appear uninitialized!")
        else:
            print(f"⚠️  Checkpoint not found at {self.model_path}")
            print("Creating new model with random weights")

        return model.to(self.device)

    def _tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize text input"""
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.max_seq_length,
        )
        return tokens["input_ids"].to(self.device)

    def _detokenize(self, tokens: torch.Tensor) -> str:
        """Convert tokens back to text"""
        # Ensure tokens are on CPU and handle different tensor shapes
        if tokens.dim() > 1:
            tokens = tokens.squeeze(0)
        tokens_cpu = tokens.cpu().numpy() if torch.is_tensor(tokens) else tokens
        return self.tokenizer.decode(tokens_cpu, skip_special_tokens=True)

    def _generate_response(
        self, prompt: str
    ) -> Tuple[str, torch.Tensor, torch.Tensor, Dict]:
        """Generate response using masked generation process"""

        # Tokenize prompt
        prompt_tokens = self._tokenize_text(prompt)
        batch_size = prompt_tokens.shape[0]

        # Initialize or reuse thought stack
        if self.thought_stack is None:
            self.thought_stack = self.model.initialize_stack(batch_size, self.device)

        # Generate using diffusion process
        print(f"   Prompt tokens shape: {prompt_tokens.shape}")
        print(
            f"   Generating with {self.num_diffusion_steps} diffusion steps, temp={self.temperature}"
        )

        # Prepare sequence for diffusion generation
        max_new_tokens = min(self.max_length, 100)
        total_seq_len = prompt_tokens.shape[1] + max_new_tokens

        # Create full sequence: known prompt + random noise for generation
        full_sequence = torch.cat(
            [
                prompt_tokens,
                torch.randint(
                    0,
                    self.model.vocab_size,
                    (batch_size, max_new_tokens),
                    device=self.device,
                ),
            ],
            dim=1,
        )

        # Create mask: True for positions to generate (after prompt)
        generation_mask = torch.zeros(
            batch_size, total_seq_len, dtype=torch.bool, device=self.device
        )
        generation_mask[:, prompt_tokens.shape[1] :] = True

        # Diffusion denoising process
        current_stack = self.thought_stack

        stats = {
            "total_unmasked": 0,
            "steps_taken": 0,
            "generation_method": "diffusion_denoising",
        }

        # Progressive denoising over multiple diffusion steps
        for step in range(self.num_diffusion_steps):
            # Create timestep for this diffusion step
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)

            # Forward pass through model
            with torch.no_grad():
                logits, _, updated_stack = self.model(
                    full_sequence,
                    current_stack,
                    timestep=t,
                    token_mask=generation_mask,
                    return_all_thoughts=True,
                )

            # Update stack
            current_stack = updated_stack

            # Apply denoising to generation positions only
            if step < self.num_diffusion_steps - 1:  # Not the final step
                # Sample with noise for intermediate steps
                next_token_logits = logits / (
                    self.temperature * (1 + step * 0.01)
                )  # Reduce temp over time
                probs = torch.softmax(next_token_logits, dim=-1)

                # Update only the generation positions
                for pos in range(prompt_tokens.shape[1], total_seq_len):
                    if torch.rand(1).item() < 0.7:  # Progressive refinement
                        new_token = torch.multinomial(probs[:, pos, :], num_samples=1)
                        full_sequence[:, pos] = new_token.squeeze(1)
                        stats["total_unmasked"] += 1
            else:
                # Final step - deterministic or low-temperature sampling
                next_token_logits = logits / self.temperature
                probs = torch.softmax(next_token_logits, dim=-1)

                for pos in range(prompt_tokens.shape[1], total_seq_len):
                    new_token = torch.multinomial(probs[:, pos, :], num_samples=1)
                    full_sequence[:, pos] = new_token.squeeze(1)

            stats["steps_taken"] = step + 1

        # Update final thought stack
        self.thought_stack = current_stack

        print(f"   Completed {stats['steps_taken']} diffusion steps")
        print(f"   Total token updates: {stats['total_unmasked']}")

        # Extract generated portion (after prompt)
        generated_tokens = full_sequence[:, prompt_tokens.shape[1] :]

        # Find EOS token and truncate
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None and generated_tokens.shape[1] > 0:
            # Find first occurrence of EOS token
            eos_positions = (generated_tokens == eos_token_id).nonzero(as_tuple=True)
            if len(eos_positions[1]) > 0:
                first_eos_pos = eos_positions[1][0].item()
                print(f"   Found EOS at position {first_eos_pos}, truncating...")
                generated_tokens = generated_tokens[:, :first_eos_pos]

        # Convert to text
        response = self._detokenize(generated_tokens)

        # Get the last thought from the stack for metrics
        # For 3D model: stack shape is (B, C, H, W, D)
        last_thought = current_stack[:, :, :, :, 0].flatten(
            1
        )  # Most recent thought, flatten spatial dims

        return response, last_thought, current_stack, stats

    def chat_turn(self, user_input: str) -> Dict:
        """Process one turn of conversation"""

        # Add user input to history
        self.conversation_history.append({"role": "user", "content": user_input})

        # Create context from conversation history
        context = ""
        for turn in self.conversation_history[-5:]:  # Use last 5 turns for context
            if turn["role"] == "user":
                context += f"Human: {turn['content']}\n"
            else:
                context += f"Assistant: {turn['content']}\n"

        context += "Assistant: "

        # Generate response
        print("🤖 Generating response...")
        print(f"   Context length: {len(context)} chars")
        response, new_thought, current_stack, gen_stats = self._generate_response(
            context
        )
        print(f"   Raw response: {repr(response[:100])}...")  # Show first 100 chars

        # Clean up response (remove prompt echo)
        response = response.strip()
        if response.startswith("Assistant:"):
            response = response[10:].strip()

        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})

        # Store thought evolution
        self.thought_history.append(
            {
                "turn": len(self.conversation_history) // 2,
                "thought_norm": float(torch.norm(new_thought).item()),
                "stack_norm": float(torch.norm(current_stack).item()),
                "user_input": user_input,
                "response": response,
                "generation_stats": gen_stats,
            }
        )

        return {
            "response": response,
            "thought_norm": float(torch.norm(new_thought).item()),
            "stack_norm": float(torch.norm(current_stack).item()),
            "conversation_length": len(self.conversation_history),
            "generation_stats": gen_stats,
        }

    def reset_conversation(self):
        """Reset conversation state"""
        self.conversation_history = []
        self.thought_stack = None
        self.thought_history = []
        print("🔄 Conversation reset")

    def save_conversation(self, filename: Optional[str] = None):
        """Save conversation and thought history"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "model_path": self.model_path,
            "config": self.config,
            "conversation_history": self.conversation_history,
            "thought_history": self.thought_history,
            "settings": {
                "max_length": self.max_length,
                "temperature": self.temperature,
                "num_diffusion_steps": self.num_diffusion_steps,
                "use_fixed_timestep": self.use_fixed_timestep,
            },
        }

        os.makedirs("outputs/conversations", exist_ok=True)
        filepath = os.path.join("outputs/conversations", filename)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"💾 Conversation saved to {filepath}")
        return filepath

    def print_thought_stats(self):
        """Print statistics about thought evolution"""
        if not self.thought_history:
            print("No thought history available")
            return

        print("\n📊 Thought Evolution Statistics:")
        for turn in self.thought_history[-5:]:  # Last 5 turns
            gen_stats = turn.get("generation_stats", {})
            stats_str = f"Turn {turn['turn']}: thought_norm={turn['thought_norm']:.3f}, stack_norm={turn['stack_norm']:.3f}"
            if gen_stats:
                stats_str += f", unmasked={gen_stats.get('total_unmasked', 0)}, steps={gen_stats.get('steps_taken', 0)}"
            print(stats_str)

    def interactive_loop(self):
        """Main interactive conversation loop"""
        print("\n" + "=" * 60)
        print("🎯 Diffusion Thought Tensor Interactive Chat")
        print("=" * 60)
        print("Commands:")
        print("  /help     - Show this help")
        print("  /reset    - Reset conversation")
        print("  /save     - Save conversation")
        print("  /stats    - Show thought statistics")
        print("  /quit     - Exit chat")
        print("  /settings - Show current model settings")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if not user_input:
                    continue

                if user_input.startswith("/"):
                    # Handle commands
                    if user_input == "/help":
                        print(
                            "Commands: /help, /reset, /save, /stats, /quit, /settings"
                        )
                    elif user_input == "/reset":
                        self.reset_conversation()
                    elif user_input == "/save":
                        self.save_conversation()
                    elif user_input == "/stats":
                        self.print_thought_stats()
                    elif user_input == "/quit":
                        print("👋 Goodbye!")
                        break
                    elif user_input == "/settings":
                        masking_config = self.config.get("masking", {})
                        thought_config = self.config.get("thought_tensor", {})
                        dream_config = self.config.get("dream", {})
                        print(f"Settings:")
                        print(
                            f"  Generation: max_length={self.max_length}, temperature={self.temperature}"
                        )
                        print(
                            f"  Masking: mask_ratio={masking_config.get('mask_ratio', 0.7)}, confidence_threshold={masking_config.get('confidence_threshold', 0.75)}"
                        )
                        print(
                            f"  Thought Stack: dims={thought_config.get('input_dims', [16, 16, 128])}, depth={thought_config.get('stack_size', 8)}"
                        )
                        print(
                            f"  DREAM: bidirectional={dream_config.get('bidirectional_attention', True)}"
                        )
                    else:
                        print("Unknown command. Type /help for available commands.")
                    continue

                # Process chat turn
                result = self.chat_turn(user_input)
                print(f"\n🤖 Assistant: {result['response']}")
                print(f"   💭 Thought norm: {result['thought_norm']:.3f}")

                # Show generation stats
                gen_stats = result.get("generation_stats", {})
                if gen_stats:
                    print(
                        f"   📋 Generation: {gen_stats.get('total_unmasked', 0)} tokens unmasked, {gen_stats.get('steps_taken', 0)} steps"
                    )

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Type /reset to reset conversation or /quit to exit")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive chat with Diffusion Thought Tensor Model"
    )
    parser.add_argument(
        "--model", "-m", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--config", "-c", type=str, help="Path to config file")
    parser.add_argument(
        "--tokenizer", "-t", type=str, default="gpt2", help="Tokenizer name"
    )
    parser.add_argument(
        "--max-length", type=int, default=100, help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument(
        "--steps", type=int, default=50, help="Number of diffusion steps"
    )
    parser.add_argument(
        "--reverse-diffusion",
        action="store_true",
        help="Use reverse diffusion process instead of fixed timestep",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.3,
        help="Token masking ratio for DREAM generation",
    )
    parser.add_argument(
        "--use-dream-config",
        action="store_true",
        help="Use dream_config.yaml as default config",
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model checkpoint not found: {args.model}")
        print("\nAvailable checkpoints:")

        # Look for checkpoints in common locations
        checkpoint_dirs = [
            "outputs/dream_checkpoints",  # New DREAM checkpoint location
            "outputs/enhanced_checkpoints",
            "outputs/simple_real_training",
            "outputs/real_data_checkpoints",
            "outputs/checkpoints",
        ]

        found_any = False
        for checkpoint_dir in checkpoint_dirs:
            if os.path.exists(checkpoint_dir):
                files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
                if files:
                    print(f"\n{checkpoint_dir}/:")
                    for f in files:
                        print(f"  {f}")
                    found_any = True

        if not found_any:
            print("  No checkpoints found in standard directories")

        sys.exit(1)

    # Handle config path defaults
    config_path = args.config
    if args.use_dream_config and not config_path:
        config_path = "configs/dream_config.yaml"

    # Create chat interface
    try:
        chat = InteractiveDiffusionChat(
            model_path=args.model,
            config_path=config_path,
            tokenizer_name=args.tokenizer,
            max_length=args.max_length,
            temperature=args.temperature,
            num_diffusion_steps=args.steps,
            use_fixed_timestep=not args.reverse_diffusion,
        )

        # Start interactive loop
        chat.interactive_loop()

    except Exception as e:
        print(f"❌ Failed to initialize chat: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
