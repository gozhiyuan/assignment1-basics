"""
Text generation utilities for Transformer language models.

This module provides functions for generating text from trained models with various
sampling strategies including temperature scaling and nucleus (top-p) sampling.
"""

import torch
import numpy as np
from typing import List, Optional, Union
from pathlib import Path

try:
    from .model import TransformerLM, softmax
    from .train_model import load_model_state_from_checkpoint
except ImportError:
    # Handle case when running as script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from cs336_basics.model import TransformerLM, softmax
    from cs336_basics.train_model import load_model_state_from_checkpoint


def temperature_scaled_softmax(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits before softmax.
    
    Args:
        logits: Raw logits tensor of shape (..., vocab_size)
        temperature: Temperature parameter τ. Lower values make distribution more peaked.
                    τ → 0: approaches argmax (deterministic)
                    τ = 1: standard softmax
                    τ > 1: more uniform distribution
    
    Returns:
        Temperature-scaled probability distribution
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    scaled_logits = logits / temperature
    return softmax(scaled_logits, dim=-1)


def nucleus_sampling(probs: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """
    Apply nucleus (top-p) sampling to a probability distribution.
    
    Args:
        probs: Probability distribution of shape (..., vocab_size)
        p: Cumulative probability threshold (0 < p <= 1)
    
    Returns:
        Modified probability distribution with low-probability tokens zeroed out
    """
    if not (0 < p <= 1):
        raise ValueError("p must be in (0, 1]")
    
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find indices where cumulative probability exceeds p
    # We want to keep tokens until cumsum >= p, so we use cumsum - sorted_probs < p
    # This ensures we include the token that pushes us over the threshold
    mask = cumulative_probs - sorted_probs < p
    
    # Set probabilities of excluded tokens to 0
    sorted_probs[~mask] = 0.0
    
    # Normalize the remaining probabilities
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    
    # Scatter back to original order
    nucleus_probs = torch.zeros_like(probs)
    nucleus_probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
    
    return nucleus_probs


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Sample next token from model logits with optional temperature and nucleus sampling.
    
    Args:
        logits: Model output logits of shape (..., vocab_size)
        temperature: Temperature for scaling (default: 1.0)
        top_p: Nucleus sampling threshold (default: None, no nucleus sampling)
        generator: Random generator for reproducible sampling
    
    Returns:
        Sampled token indices of shape (...)
    """
    # Apply temperature scaling
    probs = temperature_scaled_softmax(logits, temperature)
    
    # Apply nucleus sampling if requested
    if top_p is not None:
        probs = nucleus_sampling(probs, top_p)
    
    # Sample from the distribution
    token = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    
    return token


def generate_text(
    model: TransformerLM,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Generate text completion from a prompt using the language model.
    
    Args:
        model: Trained TransformerLM model
        prompt_tokens: Input prompt as tensor of token IDs, shape (batch_size, prompt_length)
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for sampling (default: 1.0)
        top_p: Nucleus sampling threshold (default: None)
        eos_token_id: End-of-sequence token ID to stop generation (default: None)
        device: Device to run generation on
        generator: Random generator for reproducible sampling
        
    Returns:
        Generated sequence including prompt, shape (batch_size, prompt_length + generated_length)
    """
    model.eval()
    
    # Move inputs to device
    prompt_tokens = prompt_tokens.to(device)
    batch_size, prompt_length = prompt_tokens.shape
    
    # Initialize generation sequence with prompt
    generated = prompt_tokens.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get current sequence length
            current_length = generated.shape[1]
            
            # Forward pass through model
            logits = model(generated)  # Shape: (batch_size, current_length, vocab_size)
            
            # Get logits for the last position (next token prediction)
            next_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
            
            # Sample next token
            next_token = sample_next_token(
                next_token_logits, 
                temperature=temperature, 
                top_p=top_p,
                generator=generator
            )  # Shape: (batch_size,)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            
            # Check for end-of-sequence token
            if eos_token_id is not None:
                # Stop if all sequences have generated EOS
                if (next_token == eos_token_id).all():
                    break
    
    return generated


def load_model_from_checkpoint(
    checkpoint_path: str,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float = 10000.0,
    device: str = "cpu"
) -> TransformerLM:
    """
    Load a trained model from checkpoint for generation.
    
    Args:
        checkpoint_path: Path to the saved checkpoint
        vocab_size: Model vocabulary size
        context_length: Model context length
        d_model: Model embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        rope_theta: RoPE theta parameter
        device: Device to load model on
        
    Returns:
        Loaded TransformerLM model ready for generation
    """
    # Initialize model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    ).to(device)
    
    # Load checkpoint using the function from train_model.py
    iteration = load_model_state_from_checkpoint(checkpoint_path, model, device)
    print(f"Loaded model from checkpoint at iteration {iteration}")
    
    return model


def main():
    """
    Command-line interface for text generation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate text using a trained Transformer model")
    
    # Model and checkpoint
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--vocab_size", type=int, required=True,
                        help="Model vocabulary size")
    parser.add_argument("--context_length", type=int, default=512,
                        help="Model context length")
    parser.add_argument("--d_model", type=int, default=768,
                        help="Model embedding dimension")
    parser.add_argument("--num_layers", type=int, default=12,
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=3072,
                        help="Feed-forward dimension")
    parser.add_argument("--rope_theta", type=float, default=10000.0,
                        help="RoPE theta parameter")
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt for generation")
    parser.add_argument("--prompt_tokens", type=str, default=None,
                        help="Comma-separated token IDs for prompt (alternative to --prompt)")
    parser.add_argument("--max_new_tokens", type=int, default=100,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Nucleus sampling threshold")
    parser.add_argument("--eos_token_id", type=int, default=None,
                        help="End-of-sequence token ID")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible generation")
    
    # Device
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for generation ('cpu', 'cuda', 'mps')")
    
    # Output
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    generator = None
    if args.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Load model
    print(f"Loading model from {args.checkpoint_path}...")
    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device
    )
    
    # Prepare prompt
    if args.prompt_tokens is not None:
        # Use provided token IDs
        prompt_token_ids = [int(x.strip()) for x in args.prompt_tokens.split(",")]
        prompt_tokens = torch.tensor([prompt_token_ids], dtype=torch.long)
        print(f"Using prompt tokens: {prompt_token_ids}")
    elif args.prompt:
        # For now, we'll need a tokenizer to convert text to tokens
        # This is a placeholder - in practice you'd use your trained tokenizer
        print("Warning: Text prompt provided but no tokenizer available.")
        print("Please provide token IDs using --prompt_tokens instead.")
        print(f"Text prompt was: '{args.prompt}'")
        return
    else:
        # Empty prompt (just start token or random)
        prompt_tokens = torch.tensor([[0]], dtype=torch.long)  # Assuming 0 is a valid start token
        print("Using empty prompt (token ID 0)")
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    print(f"Parameters: max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, top_p={args.top_p}")
    print("-" * 60)
    
    for i in range(args.num_samples):
        # Generate text
        generated_tokens = generate_text(
            model=model,
            prompt_tokens=prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=args.eos_token_id,
            device=args.device,
            generator=generator
        )
        
        # Extract generated tokens (remove prompt)
        prompt_length = prompt_tokens.shape[1]
        new_tokens = generated_tokens[0, prompt_length:].tolist()
        all_tokens = generated_tokens[0].tolist()
        
        print(f"Sample {i + 1}:")
        print(f"  Full sequence: {all_tokens}")
        print(f"  New tokens: {new_tokens}")
        print(f"  Length: {len(new_tokens)} new tokens")
        print()


if __name__ == "__main__":
    main()
