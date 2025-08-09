# Model Training Implementation Guide

This document provides a comprehensive guide to the Transformer language model training implementation, including data loading, checkpointing, and the complete training loop.

## Table of Contents

1. [Overview](#overview)
2. [Data Loading System](#data-loading-system)
3. [Checkpointing System](#checkpointing-system)
4. [Training Loop](#training-loop)
5. [Usage Examples](#usage-examples)
6. [Technical Details](#technical-details)
7. [Performance Considerations](#performance-considerations)

## Overview

The training system brings together all components to train Transformer language models efficiently:

- **Memory-efficient data loading** with support for large datasets
- **Robust checkpointing** for fault-tolerant training
- **Complete training loop** with learning rate scheduling, logging, and validation
- **Command-line interface** for easy experimentation with hyperparameters

## Data Loading System

### Random Sampling vs Sequential Iteration

The data loader uses **random sampling** rather than sequential iteration for several important reasons:

#### Why Random Sampling?

**Better Training Dynamics:**
- **Unbiased gradient estimates**: Random batches provide better approximations of the true gradient
- **Natural regularization**: Prevents overfitting to sequence order and document boundaries
- **Faster convergence**: More diverse batches lead to more effective learning

**Practical Benefits:**
- **Stateless**: No need to track which sequences have been seen
- **Scalable**: Works with arbitrarily large datasets
- **Industry standard**: Used by all major language models (GPT, PaLM, etc.)

#### Example Comparison

```python
# Sequential (problematic)
Batch 1: [x₀, x₁, x₂], [x₃, x₄, x₅], [x₆, x₇, x₈]
Batch 2: [x₉, x₁₀, x₁₁], [x₁₂, x₁₃, x₁₄], [x₁₅, x₁₆, x₁₇]
# → Correlated examples, predictable patterns

# Random sampling (our approach)
Batch 1: [x₄₂, x₄₃, x₄₄], [x₁₇₈, x₁₇₉, x₁₈₀], [x₉₅₆, x₉₅₇, x₉₅₈]
Batch 2: [x₃₃, x₃₄, x₃₅], [x₇₂₁, x₇₂₂, x₇₂₃], [x₁₂, x₁₃, x₁₄]
# → Diverse, uncorrelated examples
```

### Memory-Mapped Data Loading

For large datasets that don't fit in RAM, the system uses memory mapping:

```python
def load_dataset(file_path, mmap_mode=True, dtype=np.int64):
    """Load dataset with optional memory mapping."""
    if mmap_mode:
        # Data loaded lazily as accessed - saves RAM
        data = np.load(file_path, mmap_mode='r')
    else:
        # Load entire file into memory
        data = np.load(file_path)
    return data
```

**Benefits of Memory Mapping:**
- **RAM efficient**: Only loads data as needed
- **Large dataset support**: Handle datasets larger than available RAM
- **Fast access**: OS-level caching for frequently accessed regions
- **Transparent**: Works exactly like regular numpy arrays

**Usage:**
```python
# For large datasets (recommended)
train_data = load_dataset("large_dataset.npy", mmap_mode=True)

# For small datasets
train_data = load_dataset("small_dataset.npy", mmap_mode=False)
```

### Batch Generation

The `get_batch` function efficiently samples training examples:

```python
def get_batch(data, batch_size, context_length, device):
    """Sample random sequences for training."""
    # Calculate valid starting positions
    max_start_idx = len(data) - context_length - 1
    
    # Sample random starting indices
    start_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)
    
    # Create input and target sequences (targets = inputs shifted by 1)
    inputs = np.zeros((batch_size, context_length), dtype=np.int64)
    targets = np.zeros((batch_size, context_length), dtype=np.int64)
    
    for i, start_idx in enumerate(start_indices):
        inputs[i] = data[start_idx:start_idx + context_length]
        targets[i] = data[start_idx + 1:start_idx + context_length + 1]
    
    return torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)
```

## Checkpointing System

### Why Checkpointing Matters

Training large language models can take days or weeks. Checkpointing enables:

- **Fault tolerance**: Resume after system failures or job timeouts
- **Experiment tracking**: Save models at different training stages
- **Analysis**: Study training dynamics post-hoc
- **Deployment**: Export trained models for inference

### Implementation

**Saving Checkpoints:**
```python
def save_checkpoint(model, optimizer, iteration, out):
    """Save complete training state."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)
```

**Loading Checkpoints:**
```python
def load_checkpoint(src, model, optimizer):
    """Restore training state and return iteration number."""
    checkpoint = torch.load(src, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']
```

**Key Features:**
- **Complete state preservation**: Model weights, optimizer state, iteration count
- **Device agnostic**: `map_location='cpu'` ensures cross-device compatibility
- **File format flexibility**: Supports paths and file-like objects

## Training Loop

### Core Architecture

The training loop integrates all components into a complete system:

```python
def train_model(
    train_data_path,
    val_data_path=None,
    vocab_size=10000,
    context_length=512,
    d_model=768,
    num_layers=12,
    num_heads=12,
    d_ff=3072,
    # ... many more configurable parameters
):
    # 1. Load data with memory mapping
    train_data = load_dataset(train_data_path, mmap_mode=True)
    
    # 2. Initialize model and optimizer
    model = TransformerLM(vocab_size, context_length, d_model, ...)
    optimizer = AdamW(model.parameters(), lr=learning_rate, ...)
    
    # 3. Resume from checkpoint if requested
    start_iter = 0
    if resume_from_checkpoint:
        start_iter = load_checkpoint(checkpoint_path, model, optimizer)
    
    # 4. Training loop
    for iter_num in range(start_iter, max_iters):
        # Learning rate scheduling
        lr = compute_learning_rate(iter_num, warmup_iters, max_iters)
        
        # Forward pass
        inputs, targets = get_batch(train_data, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        run_gradient_clipping(model.parameters(), gradient_clip_val)
        optimizer.step()
        
        # Periodic logging, evaluation, and checkpointing
        if iter_num % log_interval == 0:
            print(f"Iter {iter_num:6d} | Loss: {loss.item():.4f} | LR: {lr:.2e}")
        
        if iter_num % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, iter_num, checkpoint_path)
```

### Learning Rate Scheduling

**Two-phase schedule:**
1. **Linear warmup**: Gradual increase from 0 to max learning rate
2. **Cosine decay**: Smooth decrease to minimum learning rate

```python
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
        0,
        cosine_iters
    )
```

### Advanced Features

**Gradient Clipping:**
```python
if gradient_clip_val > 0:
    run_gradient_clipping(model.parameters(), gradient_clip_val)
```

**Validation Evaluation:**
```python
if val_data is not None and iter_num % eval_interval == 0:
    model.eval()
    with torch.no_grad():
        val_inputs, val_targets = get_batch(val_data, batch_size, context_length, device)
        val_logits = model(val_inputs)
        val_loss = cross_entropy(val_logits.view(-1, vocab_size), val_targets.view(-1))
        print(f"Iter {iter_num:6d} | Val Loss: {val_loss.item():.4f}")
    model.train()
```

## Usage Examples

### Basic Training

Train a small model for experimentation:

```bash
uv run python cs336_basics/train_model.py \
  --train_data data/train_tokens.npy \
  --vocab_size 10000 \
  --context_length 512 \
  --d_model 256 \
  --num_layers 6 \
  --num_heads 8 \
  --batch_size 32 \
  --max_iters 50000 \
  --device cpu
```

### Large-Scale Training

Production-scale model with all features:

```bash
uv run python cs336_basics/train_model.py \
  --train_data data/large_train_tokens.npy \
  --val_data data/large_val_tokens.npy \
  --vocab_size 50000 \
  --context_length 2048 \
  --d_model 1024 \
  --num_layers 24 \
  --num_heads 16 \
  --d_ff 4096 \
  --batch_size 64 \
  --learning_rate 2e-4 \
  --min_learning_rate 1e-5 \
  --weight_decay 0.1 \
  --warmup_iters 5000 \
  --max_iters 500000 \
  --eval_interval 1000 \
  --log_interval 100 \
  --checkpoint_interval 10000 \
  --checkpoint_path checkpoints/large_model.pt \
  --device cuda \
  --gradient_clip_val 1.0
```

### Resume Training

Continue from a saved checkpoint:

```bash
uv run python cs336_basics/train_model.py \
  --train_data data/train_tokens.npy \
  --resume \
  --checkpoint_path checkpoints/model.pt \
  --max_iters 100000
```

### Apple Silicon (M1/M2/M3)

Training on Apple Silicon with MPS acceleration:

```bash
uv run python cs336_basics/train_model.py \
  --train_data data/train_tokens.npy \
  --device mps \
  --batch_size 16 \
  --context_length 1024
```

## Technical Details

### Command-Line Arguments

**Data Arguments:**
- `--train_data`: Path to training data (required)
- `--val_data`: Path to validation data (optional)

**Model Architecture:**
- `--vocab_size`: Vocabulary size (default: 10000)
- `--context_length`: Maximum sequence length (default: 512)
- `--d_model`: Model embedding dimension (default: 768)
- `--num_layers`: Number of transformer layers (default: 12)
- `--num_heads`: Number of attention heads (default: 12)
- `--d_ff`: Feed-forward hidden dimension (default: 3072)
- `--rope_theta`: RoPE theta parameter (default: 10000.0)

**Training Hyperparameters:**
- `--batch_size`: Training batch size (default: 32)
- `--learning_rate`: Maximum learning rate (default: 1e-4)
- `--min_learning_rate`: Minimum learning rate (default: 1e-5)
- `--weight_decay`: AdamW weight decay (default: 0.01)
- `--warmup_iters`: Warmup iterations (default: 2000)
- `--max_iters`: Maximum iterations (default: 100000)
- `--gradient_clip_val`: Gradient clipping threshold (default: 1.0)

**Logging and Checkpointing:**
- `--eval_interval`: Validation frequency (default: 1000)
- `--log_interval`: Logging frequency (default: 100)
- `--checkpoint_interval`: Checkpoint frequency (default: 5000)
- `--checkpoint_path`: Checkpoint file path (default: "checkpoint.pt")
- `--resume`: Resume from existing checkpoint
- `--device`: Training device ('cpu', 'cuda', 'mps')

### Model Integration

The training loop uses these key components:

**From `cs336_basics.model`:**
- `TransformerLM`: Complete transformer language model with RoPE

**From `cs336_basics.optimizer`:**
- `AdamW`: Optimizer with decoupled weight decay
- `cross_entropy`: Numerically stable loss computation
- `run_gradient_clipping`: Gradient norm clipping
- `run_get_lr_cosine_schedule`: Cosine learning rate scheduling

**From `cs336_basics.train_model`:**
- `get_batch`: Random batch sampling
- `load_dataset`: Memory-mapped data loading
- `save_checkpoint` / `load_checkpoint`: Training state persistence

### Expected Output

```
============================================================
TRANSFORMER LANGUAGE MODEL TRAINING
============================================================
Loading training data from data/train_tokens.npy...
Loaded dataset: 10,000,000 tokens, dtype: int64, memory-mapped: True
Token range: 0 to 49999
Loading validation data from data/val_tokens.npy...
Loaded dataset: 1,000,000 tokens, dtype: int64, memory-mapped: True
Token range: 0 to 49999
Initializing TransformerLM with 12 layers, 768 dimensions...
Model parameters: 124,439,808 total, 124,439,808 trainable
Starting training from iteration 0 to 100000...
Device: cuda
Batch size: 32, Context length: 512
------------------------------------------------------------
Iter      0 | Loss: 10.8234 | LR: 5.00e-08
Iter    100 | Loss: 8.2156 | LR: 5.00e-06
Iter    200 | Loss: 7.1432 | LR: 1.00e-05
Iter    500 | Loss: 6.2341 | LR: 2.50e-05
Iter   1000 | Loss: 5.4123 | LR: 5.00e-05 | Val Loss: 5.6234
Iter   1500 | Loss: 4.8765 | LR: 7.50e-05
Iter   2000 | Loss: 4.3456 | LR: 1.00e-04
Iter   2100 | Loss: 4.2987 | LR: 9.95e-05
...
Iter   5000 | Loss: 3.1234 | LR: 8.77e-05 | Val Loss: 3.2456
Saving checkpoint at iteration 5000...
...
Training complete! Saving final checkpoint...
Final checkpoint saved to checkpoint.pt
```

## Performance Considerations

### Memory Optimization

**Memory-Mapped Data Loading:**
- Use `mmap_mode=True` for datasets larger than available RAM
- OS handles caching and memory management automatically
- Significantly reduces memory footprint for large datasets

**Batch Size Selection:**
- Larger batches: Better GPU utilization, more stable gradients
- Smaller batches: Less memory usage, more frequent updates
- Rule of thumb: Largest batch size that fits in memory

**Context Length Trade-offs:**
- Longer contexts: Better long-range modeling, quadratic memory growth
- Shorter contexts: Lower memory usage, faster training
- Sweet spot depends on available memory and task requirements

### Training Speed

**Device Selection:**
- **CUDA**: Best performance for NVIDIA GPUs
- **MPS**: Good performance for Apple Silicon (M1/M2/M3)
- **CPU**: Fallback option, much slower but always available

**Gradient Accumulation:**
If memory is limited, implement gradient accumulation:
```python
# Accumulate gradients over multiple mini-batches
effective_batch_size = 64
mini_batch_size = 16
accumulation_steps = effective_batch_size // mini_batch_size

for step in range(accumulation_steps):
    inputs, targets = get_batch(data, mini_batch_size, context_length, device)
    logits = model(inputs)
    loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    loss = loss / accumulation_steps  # Scale loss
    loss.backward()

optimizer.step()
optimizer.zero_grad()
```

### Hyperparameter Tuning

**Learning Rate:**
- Start with 1e-4 for most models
- Scale with batch size: `lr = base_lr * sqrt(batch_size / base_batch_size)`
- Use learning rate finder for optimal values

**Warmup Duration:**
- Typical: 2000-10000 iterations
- Larger models benefit from longer warmup
- Should be 1-10% of total training iterations

**Weight Decay:**
- Standard values: 0.01-0.1
- Higher for larger models to prevent overfitting
- Balance between regularization and model capacity

This comprehensive training system provides everything needed to train state-of-the-art Transformer language models efficiently and reliably!
