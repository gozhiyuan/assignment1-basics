import numpy as np
import torch
import os
from pathlib import Path
from typing import BinaryIO, IO


def get_batch(
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of input sequences and their corresponding targets from a dataset.
    
    Args:
        data: 1D numpy array of integer token IDs (can be memory-mapped)
        batch_size: Number of sequences to sample in the batch
        context_length: Length of each sequence
        device: PyTorch device string (e.g., 'cpu', 'cuda:0', 'mps')
        
    Returns:
        Tuple of (inputs, targets) where:
        - inputs: tensor of shape (batch_size, context_length) with input sequences
        - targets: tensor of shape (batch_size, context_length) with target sequences
                  (inputs shifted by 1 position)
    """
    # Ensure we have enough data to sample sequences of the required length
    assert len(data) > context_length, f"Dataset length {len(data)} must be greater than context_length {context_length}"
    
    # Calculate valid starting positions (we need context_length + 1 tokens to get input and target)
    max_start_idx = len(data) - context_length - 1
    
    # Sample random starting indices for each sequence in the batch
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
    
    # Create input and target sequences
    inputs = np.zeros((batch_size, context_length), dtype=np.int64)
    targets = np.zeros((batch_size, context_length), dtype=np.int64)
    
    for i, start_idx in enumerate(start_indices):
        inputs[i] = data[start_idx:start_idx + context_length]
        targets[i] = data[start_idx + 1:start_idx + context_length + 1]
    
    # Convert to PyTorch tensors and move to specified device
    inputs_tensor = torch.from_numpy(inputs).to(device)
    targets_tensor = torch.from_numpy(targets).to(device)
    
    return inputs_tensor, targets_tensor


def load_dataset(
    file_path: str | Path, 
    mmap_mode: bool = True,
    dtype: np.dtype = np.int64
) -> np.ndarray:
    """
    Load a dataset with optional memory mapping for large files.
    
    Args:
        file_path: Path to the dataset file (.npy or .npz)
        mmap_mode: If True, use memory mapping to avoid loading entire file into RAM
        dtype: Expected data type of the array
        
    Returns:
        numpy array (potentially memory-mapped) containing the dataset
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.npy':
        if mmap_mode:
            # Use memory mapping - data is loaded lazily as accessed
            data = np.load(file_path, mmap_mode='r')
            # Verify the data type matches expectations
            if data.dtype != dtype:
                print(f"Warning: Expected dtype {dtype}, got {data.dtype}")
        else:
            # Load entire file into memory
            data = np.load(file_path)
    elif file_path.suffix == '.npz':
        # For .npz files, we need to specify the array name
        # This will load into memory (npz doesn't support mmap directly)
        npz_data = np.load(file_path)
        # Assume the data is stored under 'data' key, adjust as needed
        data = npz_data['data'] if 'data' in npz_data else npz_data[list(npz_data.keys())[0]]
    else:
        # For other formats, try memory mapping directly
        if mmap_mode:
            data = np.memmap(file_path, dtype=dtype, mode='r')
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Verify the data looks reasonable
    if len(data) == 0:
        raise ValueError("Dataset is empty")
    
    print(f"Loaded dataset: {len(data):,} tokens, dtype: {data.dtype}, memory-mapped: {mmap_mode}")
    
    # Sanity check - ensure token IDs are reasonable
    if len(data) > 0:
        min_token, max_token = data.min(), data.max()
        print(f"Token range: {min_token} to {max_token}")
    
    return data


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]
) -> None:
    """
    Save model and optimizer state to a checkpoint file.
    
    Args:
        model: PyTorch model to save
        optimizer: PyTorch optimizer to save
        iteration: Current training iteration number
        out: Path or file-like object to save checkpoint to
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str | None = None
) -> int:
    """
    Load model and optimizer state from a checkpoint file.
    
    Args:
        src: Path or file-like object to load checkpoint from
        model: PyTorch model to restore state to
        optimizer: PyTorch optimizer to restore state to
        device: Device to load checkpoint on (if None, uses CPU for safety)
        
    Returns:
        int: The iteration number from the checkpoint
    """
    # Use CPU as default for cross-platform compatibility, but allow override
    map_location = 'cpu' if device is None else device
    checkpoint = torch.load(src, map_location=map_location)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['iteration']


def load_model_state_from_checkpoint(
    checkpoint_path: str | os.PathLike,
    model: torch.nn.Module,
    device: str = "cpu"
) -> int:
    """
    Load only model state from checkpoint into an existing model (for generation/inference).
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model to restore state to
        device: Device to load model on
        
    Returns:
        int: The iteration number from the checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint['iteration']


def train_model(
    train_data_path: str,
    val_data_path: str | None = None,
    vocab_size: int = 10000,
    context_length: int = 512,
    d_model: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    d_ff: int = 3072,
    rope_theta: float = 10000.0,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    min_learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    warmup_iters: int = 2000,
    max_iters: int = 100000,
    eval_interval: int = 1000,
    log_interval: int = 100,
    checkpoint_interval: int = 5000,
    checkpoint_path: str = "checkpoint.pt",
    device: str = "cpu",
    resume_from_checkpoint: bool = False,
    gradient_clip_val: float = 1.0,
) -> None:
    """
    Train a Transformer language model.
    
    Args:
        train_data_path: Path to training data (tokenized numpy array)
        val_data_path: Path to validation data (optional)
        vocab_size: Size of the vocabulary
        context_length: Maximum sequence length
        d_model: Model embedding dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        rope_theta: RoPE theta parameter
        batch_size: Training batch size
        learning_rate: Maximum learning rate
        min_learning_rate: Minimum learning rate (for cosine schedule)
        weight_decay: Weight decay for AdamW
        warmup_iters: Number of warmup iterations
        max_iters: Maximum number of training iterations
        eval_interval: Evaluate on validation set every N iterations
        log_interval: Log training metrics every N iterations
        checkpoint_interval: Save checkpoint every N iterations
        checkpoint_path: Path to save/load checkpoints
        device: Device to train on ('cpu', 'cuda', 'mps')
        resume_from_checkpoint: Whether to resume from existing checkpoint
        gradient_clip_val: Maximum gradient norm for clipping
    """
    # Import required classes
    from cs336_basics.model import TransformerLM
    from cs336_basics.optimizer import AdamW, cross_entropy, run_gradient_clipping, run_get_lr_cosine_schedule
    
    print("=" * 60)
    print("TRANSFORMER LANGUAGE MODEL TRAINING")
    print("=" * 60)
    
    # Load datasets with memory mapping
    print(f"Loading training data from {train_data_path}...")
    train_data = load_dataset(train_data_path, mmap_mode=True)
    
    val_data = None
    if val_data_path:
        print(f"Loading validation data from {val_data_path}...")
        val_data = load_dataset(val_data_path, mmap_mode=True)
    
    # Initialize model
    print(f"Initializing TransformerLM with {num_layers} layers, {d_model} dimensions...")
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # Setup for resuming training
    start_iter = 0
    if resume_from_checkpoint and os.path.exists(checkpoint_path):
        print(f"Resuming training from {checkpoint_path}...")
        start_iter = load_checkpoint(checkpoint_path, model, optimizer, device)
        print(f"Resumed from iteration {start_iter}")
    
    # Training loop
    print(f"Starting training from iteration {start_iter} to {max_iters}...")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, Context length: {context_length}")
    print("-" * 60)
    
    model.train()
    
    for iter_num in range(start_iter, max_iters):
        # Get learning rate for this iteration
        if iter_num < warmup_iters:
            # Linear warmup
            lr = learning_rate * (iter_num + 1) / warmup_iters
        else:
            # Cosine decay
            cosine_iters = max_iters - warmup_iters
            lr = run_get_lr_cosine_schedule(
                iter_num - warmup_iters,
                learning_rate,
                min_learning_rate,
                0,  # No additional warmup in cosine phase
                cosine_iters
            )
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass
        inputs, targets = get_batch(train_data, batch_size, context_length, device)
        
        optimizer.zero_grad()
        logits = model(inputs)  # Shape: (batch_size, context_length, vocab_size)
        
        # Compute loss
        # Reshape for cross-entropy: (batch_size * context_length, vocab_size) and (batch_size * context_length,)
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        loss = cross_entropy(logits_flat, targets_flat)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if gradient_clip_val > 0:
            run_gradient_clipping(model.parameters(), gradient_clip_val)
        
        optimizer.step()
        
        # Logging
        if iter_num % log_interval == 0:
            print(f"Iter {iter_num:6d} | Loss: {loss.item():.4f} | LR: {lr:.2e}")
        
        # Evaluation
        if val_data is not None and iter_num % eval_interval == 0 and iter_num > 0:
            model.eval()
            with torch.no_grad():
                val_inputs, val_targets = get_batch(val_data, batch_size, context_length, device)
                val_logits = model(val_inputs)
                val_logits_flat = val_logits.view(-1, vocab_size)
                val_targets_flat = val_targets.view(-1)
                val_loss = cross_entropy(val_logits_flat, val_targets_flat)
                print(f"Iter {iter_num:6d} | Val Loss: {val_loss.item():.4f}")
            model.train()
        
        # Checkpointing
        if iter_num % checkpoint_interval == 0 and iter_num > 0:
            print(f"Saving checkpoint at iteration {iter_num}...")
            save_checkpoint(model, optimizer, iter_num, checkpoint_path)
    
    # Final checkpoint
    print("Training complete! Saving final checkpoint...")
    save_checkpoint(model, optimizer, max_iters, checkpoint_path)
    print(f"Final checkpoint saved to {checkpoint_path}")


def main():
    """
    Main training script with command-line argument parsing.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data (tokenized numpy array)")
    parser.add_argument("--val_data", type=str, default=None,
                        help="Path to validation data (optional)")
    
    # Model architecture
    parser.add_argument("--vocab_size", type=int, default=10000,
                        help="Vocabulary size")
    parser.add_argument("--context_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--d_model", type=int, default=768,
                        help="Model embedding dimension")
    parser.add_argument("--num_layers", type=int, default=12,
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=3072,
                        help="Feed-forward hidden dimension")
    parser.add_argument("--rope_theta", type=float, default=10000.0,
                        help="RoPE theta parameter")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Maximum learning rate")
    parser.add_argument("--min_learning_rate", type=float, default=1e-5,
                        help="Minimum learning rate (for cosine schedule)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    parser.add_argument("--warmup_iters", type=int, default=2000,
                        help="Number of warmup iterations")
    parser.add_argument("--max_iters", type=int, default=100000,
                        help="Maximum number of training iterations")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0,
                        help="Maximum gradient norm for clipping")
    
    # Logging and checkpointing
    parser.add_argument("--eval_interval", type=int, default=1000,
                        help="Evaluate on validation set every N iterations")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Log training metrics every N iterations")
    parser.add_argument("--checkpoint_interval", type=int, default=5000,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt",
                        help="Path to save/load checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from existing checkpoint")
    
    # Device
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to train on ('cpu', 'cuda', 'mps')")
    
    args = parser.parse_args()
    
    # Validate arguments
    assert args.d_model % args.num_heads == 0, "d_model must be divisible by num_heads"
    assert args.vocab_size > 0, "vocab_size must be positive"
    assert args.context_length > 0, "context_length must be positive"
    
    # Start training
    train_model(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        warmup_iters=args.warmup_iters,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        resume_from_checkpoint=args.resume,
        gradient_clip_val=args.gradient_clip_val,
    )


if __name__ == "__main__":
    main()