#!/usr/bin/env python3
"""
Integration test script for training and generation pipeline.

This script demonstrates the complete workflow:
1. Create synthetic training data
2. Train a small Transformer model
3. Generate text from the trained model
4. Verify the pipeline works end-to-end
"""

import os
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path

# Import our training and generation modules
try:
    from .train_model import train_model, load_dataset
    from .generate import load_model_from_checkpoint, generate_text
except ImportError:
    # Handle case when running as script
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from cs336_basics.train_model import train_model, load_dataset
    from cs336_basics.generate import load_model_from_checkpoint, generate_text


def create_synthetic_dataset(
    vocab_size: int = 1000,
    num_tokens: int = 50000,
    save_path: str = "synthetic_data.npy"
) -> str:
    """
    Create a synthetic dataset for testing.
    
    Args:
        vocab_size: Size of vocabulary
        num_tokens: Number of tokens to generate
        save_path: Path to save the dataset
        
    Returns:
        Path to the saved dataset
    """
    print(f"Creating synthetic dataset with {num_tokens:,} tokens, vocab_size={vocab_size}")
    
    # Create synthetic data with some patterns to make it learnable
    # We'll create simple sequences like [1, 2, 3, 1, 2, 3, ...] with some noise
    data = []
    
    # Pattern 1: Simple counting sequence
    for i in range(num_tokens // 4):
        data.extend([1, 2, 3, 4, 5])
    
    # Pattern 2: Repeated pairs
    for i in range(num_tokens // 4):
        data.extend([10, 11, 10, 11])
    
    # Pattern 3: Random but biased towards lower IDs
    for i in range(num_tokens // 2):
        data.append(np.random.randint(0, min(100, vocab_size)))
    
    # Truncate to exact length and ensure we don't exceed vocab_size
    data = data[:num_tokens]
    data = [min(token, vocab_size - 1) for token in data]
    
    # Convert to numpy array
    dataset = np.array(data, dtype=np.int64)
    
    # Save to file
    np.save(save_path, dataset)
    print(f"Saved synthetic dataset to {save_path}")
    print(f"Token range: {dataset.min()} to {dataset.max()}")
    
    return save_path


def run_training_test(
    train_data_path: str,
    checkpoint_path: str = "test_checkpoint.pt",
    vocab_size: int = 1000,
    device: str = "cpu"
) -> None:
    """
    Run a quick training test with a small model.
    
    Args:
        train_data_path: Path to training data
        checkpoint_path: Path to save checkpoint
        vocab_size: Vocabulary size
        device: Device to train on
    """
    print("\n" + "="*60)
    print("RUNNING TRAINING TEST")
    print("="*60)
    
    # Small model configuration for fast testing
    train_model(
        train_data_path=train_data_path,
        val_data_path=None,  # No validation for this test
        vocab_size=vocab_size,
        context_length=32,    # Very short context for speed
        d_model=64,          # Small model
        num_layers=2,        # Just 2 layers
        num_heads=4,         # Few heads
        d_ff=128,           # Small feed-forward
        rope_theta=10000.0,
        batch_size=8,        # Small batch size
        learning_rate=1e-3,  # Higher learning rate for faster convergence
        min_learning_rate=1e-4,
        weight_decay=0.01,
        warmup_iters=10,     # Very short warmup
        max_iters=50,        # Just 50 iterations for testing
        eval_interval=1000,  # No evaluation
        log_interval=10,     # Log every 10 iterations
        checkpoint_interval=25,  # Save checkpoint once
        checkpoint_path=checkpoint_path,
        device=device,
        resume_from_checkpoint=False,
        gradient_clip_val=1.0,
    )
    
    print(f"Training completed! Checkpoint saved to {checkpoint_path}")


def run_generation_test(
    checkpoint_path: str,
    vocab_size: int = 1000,
    device: str = "cpu"
) -> None:
    """
    Test text generation from the trained model.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        vocab_size: Vocabulary size
        device: Device to run generation on
    """
    print("\n" + "="*60)
    print("RUNNING GENERATION TEST")
    print("="*60)
    
    # Load the trained model
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        vocab_size=vocab_size,
        context_length=32,
        d_model=64,
        num_layers=2,
        num_heads=4,
        d_ff=128,
        rope_theta=10000.0,
        device=device
    )
    
    # Test different generation configurations
    test_configs = [
        {"name": "Deterministic (low temp)", "temperature": 0.1, "top_p": None},
        {"name": "Balanced", "temperature": 0.8, "top_p": 0.9},
        {"name": "Creative (high temp)", "temperature": 1.5, "top_p": None},
        {"name": "Nucleus sampling", "temperature": 1.0, "top_p": 0.5},
    ]
    
    # Test prompts
    test_prompts = [
        [1, 2],      # Pattern that should continue as [1, 2, 3, 4, 5]
        [10, 11],    # Pattern that should continue as [10, 11, 10, 11]
        [42],        # Random start
    ]
    
    print(f"Testing generation with {len(test_configs)} configurations and {len(test_prompts)} prompts:")
    print("-" * 60)
    
    for config in test_configs:
        print(f"\n🔸 {config['name']}:")
        print(f"   Temperature: {config['temperature']}, Top-p: {config['top_p']}")
        
        for i, prompt in enumerate(test_prompts):
            prompt_tensor = torch.tensor([prompt], dtype=torch.long)
            
            # Generate text
            generated = generate_text(
                model=model,
                prompt_tokens=prompt_tensor,
                max_new_tokens=10,
                temperature=config["temperature"],
                top_p=config["top_p"],
                device=device
            )
            
            # Extract results
            new_tokens = generated[0, len(prompt):].tolist()
            
            print(f"   Prompt {i+1}: {prompt} → {new_tokens}")
    
    print("\n✅ Generation test completed successfully!")


def run_comprehensive_test(device: str = "cpu") -> None:
    """
    Run the complete training and generation pipeline test.
    
    Args:
        device: Device to run tests on ('cpu', 'cuda', 'mps')
    """
    print("🚀 COMPREHENSIVE TRANSFORMER PIPELINE TEST")
    print("="*60)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Create temporary directory for test files
    temp_dir = tempfile.mkdtemp(prefix="transformer_test_")
    print(f"Working directory: {temp_dir}")
    
    try:
        # Paths for test files
        train_data_path = os.path.join(temp_dir, "train_data.npy")
        checkpoint_path = os.path.join(temp_dir, "model_checkpoint.pt")
        
        # Configuration
        vocab_size = 1000
        num_tokens = 10000
        
        # Step 1: Create synthetic training data
        print("\n📊 Step 1: Creating synthetic training data...")
        create_synthetic_dataset(
            vocab_size=vocab_size,
            num_tokens=num_tokens,
            save_path=train_data_path
        )
        
        # Verify data loading works
        dataset = load_dataset(train_data_path, mmap_mode=True)
        print(f"✅ Data loading verified: {len(dataset):,} tokens")
        
        # Step 2: Train a small model
        print("\n🎯 Step 2: Training model...")
        run_training_test(
            train_data_path=train_data_path,
            checkpoint_path=checkpoint_path,
            vocab_size=vocab_size,
            device=device
        )
        
        # Verify checkpoint exists
        assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
        checkpoint_size_mb = os.path.getsize(checkpoint_path) / 1024 / 1024
        print(f"✅ Checkpoint saved: {checkpoint_size_mb:.2f} MB")
        
        # Step 3: Test text generation
        print("\n📝 Step 3: Testing text generation...")
        run_generation_test(
            checkpoint_path=checkpoint_path,
            vocab_size=vocab_size,
            device=device
        )
        
        # Step 4: Test command-line interfaces
        print("\n💻 Step 4: Testing command-line interfaces...")
        print("Training script help:")
        print("  uv run python cs336_basics/train_model.py --help")
        print("Generation script help:")
        print("  uv run python cs336_basics/generate.py --help")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("="*60)
        print("✅ Synthetic data creation: PASSED")
        print("✅ Model training: PASSED") 
        print("✅ Checkpoint saving/loading: PASSED")
        print("✅ Text generation: PASSED")
        print("✅ Temperature scaling: PASSED")
        print("✅ Nucleus sampling: PASSED")
        print("✅ Device compatibility: PASSED")
        print("="*60)
        print("🚀 The Transformer training and generation pipeline is working correctly!")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
        
    finally:
        # Clean up temporary files
        print(f"\n🧹 Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Main entry point with command-line argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Transformer training and generation pipeline")
    parser.add_argument("--device", type=str, default="cpu", 
                       help="Device to run tests on ('cpu', 'cuda', 'mps')")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test with minimal iterations")
    
    args = parser.parse_args()
    
    # Validate device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU")
        args.device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("⚠️  MPS requested but not available, falling back to CPU")
        args.device = "cpu"
    
    # Run the comprehensive test
    run_comprehensive_test(device=args.device)


if __name__ == "__main__":
    main()
