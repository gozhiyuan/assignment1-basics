# Text Generation Implementation Guide

This document provides a comprehensive guide to text generation from trained Transformer language models, including sampling strategies, decoding techniques, and practical usage examples.

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Background](#theoretical-background)
3. [Sampling Strategies](#sampling-strategies)
4. [Implementation Details](#implementation-details)
5. [Usage Examples](#usage-examples)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)

## Overview

Text generation is the process of using a trained language model to produce new text sequences. Our implementation provides:

- **Autoregressive generation**: Sequentially predict and sample next tokens
- **Temperature scaling**: Control randomness in sampling
- **Nucleus (top-p) sampling**: Focus on high-probability tokens
- **Flexible prompting**: Start generation from any prompt
- **Batch generation**: Generate multiple samples simultaneously

## Theoretical Background

### Language Model Output

A language model takes an integer sequence of length `sequence_length` and produces a matrix of size `(sequence_length × vocab_size)`, where each row represents a probability distribution over the vocabulary for predicting the next token.

### Decoding Process

For a sequence x₁...t, the next token xt+1 is sampled according to:

```
P(xt+1 = i | x₁...t) = exp(vᵢ) / Σⱼ exp(vⱼ)
```

where `v = TransformerLM(x₁...t)t ∈ ℝ^vocab_size` is the model output at position t.

### Basic Autoregressive Generation

1. **Input**: Prompt tokens x₁...t
2. **Forward Pass**: Compute logits v = model(x₁...t)
3. **Extract**: Take last position logits v_t for next token prediction  
4. **Sample**: Draw token xt+1 from probability distribution
5. **Append**: Add xt+1 to sequence and repeat until stopping condition

## Sampling Strategies

### 1. Temperature Scaling

Temperature scaling modifies the softmax distribution to control randomness:

```
softmax(v, τ)ᵢ = exp(vᵢ/τ) / Σⱼ exp(vⱼ/τ)
```

**Temperature Effects:**
- **τ → 0**: Approaches argmax (deterministic, picks highest probability)
- **τ = 1**: Standard softmax (unchanged distribution)
- **τ > 1**: More uniform distribution (increased randomness)

```python
def temperature_scaled_softmax(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Apply temperature scaling to logits before softmax."""
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    scaled_logits = logits / temperature
    return softmax(scaled_logits, dim=-1)
```

**Example:**
```python
logits = torch.tensor([[1.0, 2.0, 3.0, 0.5]])

# Sharp distribution (more deterministic)
probs_cold = temperature_scaled_softmax(logits, temperature=0.1)
# → [[2.06e-09, 4.54e-05, 9.99e-01, 1.39e-11]]

# Normal distribution  
probs_normal = temperature_scaled_softmax(logits, temperature=1.0)
# → [[0.0854, 0.2321, 0.6308, 0.0518]]

# Smooth distribution (more random)
probs_hot = temperature_scaled_softmax(logits, temperature=2.0)
# → [[0.1627, 0.2683, 0.4423, 0.1267]]
```

### 2. Nucleus (Top-p) Sampling

Nucleus sampling truncates low-probability tokens to focus on the most likely candidates:

```
P(xt+1 = i|q) = {
    qᵢ / Σⱼ∈V(p) qⱼ  if i ∈ V(p)
    0                otherwise
}
```

where V(p) is the smallest set of indices such that Σⱼ∈V(p) qⱼ ≥ p.

**Algorithm:**
1. Sort vocabulary by probability (descending)
2. Find smallest set where cumulative probability ≥ p
3. Zero out probabilities for excluded tokens
4. Renormalize remaining probabilities

```python
def nucleus_sampling(probs: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """Apply nucleus (top-p) sampling to probability distribution."""
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    
    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find tokens to keep (cumulative probability threshold)
    mask = cumulative_probs - sorted_probs < p
    
    # Zero out excluded tokens and renormalize
    sorted_probs[~mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    
    # Scatter back to original order
    nucleus_probs = torch.zeros_like(probs)
    nucleus_probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
    
    return nucleus_probs
```

**Example:**
```python
probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])

# Keep tokens until 80% probability mass
nucleus_80 = nucleus_sampling(probs, p=0.8)
# → [[0.625, 0.375, 0.0, 0.0]]  (top 2 tokens)

# Keep tokens until 50% probability mass  
nucleus_50 = nucleus_sampling(probs, p=0.5)
# → [[1.0, 0.0, 0.0, 0.0]]  (only top token)
```

## Implementation Details

### Core Generation Function

```python
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
    """Generate text completion from a prompt."""
    model.eval()
    
    # Move inputs to device
    prompt_tokens = prompt_tokens.to(device)
    batch_size, prompt_length = prompt_tokens.shape
    
    # Initialize generation sequence with prompt
    generated = prompt_tokens.clone()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass through model
            logits = model(generated)
            
            # Get logits for next token prediction (last position)
            next_token_logits = logits[:, -1, :]
            
            # Sample next token with temperature and nucleus sampling
            next_token = sample_next_token(
                next_token_logits, 
                temperature=temperature, 
                top_p=top_p,
                generator=generator
            )
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            
            # Check for end-of-sequence token
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
    
    return generated
```

### Sampling Function

```python
def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Sample next token from logits with optional temperature and nucleus sampling."""
    # Apply temperature scaling
    probs = temperature_scaled_softmax(logits, temperature)
    
    # Apply nucleus sampling if requested
    if top_p is not None:
        probs = nucleus_sampling(probs, top_p)
    
    # Sample from the distribution
    token = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
    
    return token
```

### Model Loading

```python
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
    """Load trained model from checkpoint for generation."""
    # Initialize model architecture
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    ).to(device)
    
    # Load weights from checkpoint
    from .optimizer import AdamW
    dummy_optimizer = AdamW(model.parameters())
    iteration = load_checkpoint(checkpoint_path, model, dummy_optimizer)
    
    return model
```

## Usage Examples

### Basic Generation

Generate text using token IDs:

```bash
uv run python cs336_basics/generate.py \
  --checkpoint_path checkpoints/model.pt \
  --vocab_size 10000 \
  --prompt_tokens "1,2,3" \
  --max_new_tokens 50 \
  --device cpu
```

### Temperature Sampling

Control randomness with temperature:

```bash
# Deterministic (low temperature)
uv run python cs336_basics/generate.py \
  --checkpoint_path checkpoints/model.pt \
  --vocab_size 10000 \
  --prompt_tokens "1,2,3" \
  --temperature 0.1 \
  --max_new_tokens 50

# Creative (high temperature)  
uv run python cs336_basics/generate.py \
  --checkpoint_path checkpoints/model.pt \
  --vocab_size 10000 \
  --prompt_tokens "1,2,3" \
  --temperature 1.5 \
  --max_new_tokens 50
```

### Nucleus Sampling

Focus on high-probability tokens:

```bash
# Conservative (top 50% of probability mass)
uv run python cs336_basics/generate.py \
  --checkpoint_path checkpoints/model.pt \
  --vocab_size 10000 \
  --prompt_tokens "1,2,3" \
  --top_p 0.5 \
  --max_new_tokens 50

# Balanced (top 90% of probability mass)
uv run python cs336_basics/generate.py \
  --checkpoint_path checkpoints/model.pt \
  --vocab_size 10000 \
  --prompt_tokens "1,2,3" \
  --top_p 0.9 \
  --max_new_tokens 50
```

### Combined Sampling

Use both temperature and nucleus sampling:

```bash
uv run python cs336_basics/generate.py \
  --checkpoint_path checkpoints/model.pt \
  --vocab_size 10000 \
  --prompt_tokens "1,2,3" \
  --temperature 0.8 \
  --top_p 0.9 \
  --max_new_tokens 100 \
  --seed 42
```

### Multiple Samples

Generate several samples for comparison:

```bash
uv run python cs336_basics/generate.py \
  --checkpoint_path checkpoints/model.pt \
  --vocab_size 10000 \
  --prompt_tokens "1,2,3" \
  --temperature 1.0 \
  --top_p 0.9 \
  --max_new_tokens 50 \
  --num_samples 5 \
  --seed 42
```

### Programmatic Usage

Use generation functions in your Python code:

```python
import torch
from cs336_basics.generate import load_model_from_checkpoint, generate_text

# Load model
model = load_model_from_checkpoint(
    checkpoint_path="checkpoints/model.pt",
    vocab_size=10000,
    context_length=512,
    d_model=768,
    num_layers=12,
    num_heads=12,
    d_ff=3072,
    device="cuda"
)

# Prepare prompt
prompt_tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)

# Generate text
generated = generate_text(
    model=model,
    prompt_tokens=prompt_tokens,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
    device="cuda"
)

# Extract new tokens
new_tokens = generated[0, prompt_tokens.shape[1]:].tolist()
print("Generated tokens:", new_tokens)
```

## Advanced Features

### Reproducible Generation

Use seeds for consistent results:

```python
# Set seed for reproducible generation
generator = torch.Generator()
generator.manual_seed(42)
torch.manual_seed(42)

generated = generate_text(
    model=model,
    prompt_tokens=prompt_tokens,
    max_new_tokens=50,
    generator=generator
)
```

### Batch Generation

Generate multiple sequences simultaneously:

```python
# Multiple prompts in batch
prompt_tokens = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
], dtype=torch.long)

# Generate for all prompts at once
generated = generate_text(
    model=model,
    prompt_tokens=prompt_tokens,
    max_new_tokens=50,
    temperature=0.8
)

# Extract results for each prompt
for i in range(prompt_tokens.shape[0]):
    new_tokens = generated[i, prompt_tokens.shape[1]:].tolist()
    print(f"Sample {i+1}: {new_tokens}")
```

### Early Stopping

Stop generation when end-of-sequence token is encountered:

```python
# Assuming token ID 50256 is <|endoftext|>
generated = generate_text(
    model=model,
    prompt_tokens=prompt_tokens,
    max_new_tokens=200,
    eos_token_id=50256,
    temperature=0.8
)
```

## Best Practices

### Sampling Strategy Selection

**Temperature Guidelines:**
- **0.1-0.5**: Focused, coherent text (good for factual content)
- **0.6-1.0**: Balanced creativity and coherence (general use)
- **1.1-2.0**: Creative, diverse text (artistic applications)

**Top-p Guidelines:**
- **0.1-0.5**: Very focused sampling (deterministic-like)
- **0.6-0.9**: Balanced sampling (recommended range)
- **0.95-1.0**: Diverse sampling (includes low-probability tokens)

**Combined Sampling:**
- Use both temperature and top-p for best results
- Temperature=0.8, top_p=0.9 is a good starting point
- Adjust based on your specific use case and quality requirements

### Performance Optimization

**Memory Management:**
- Use `torch.no_grad()` during generation to save memory
- Consider smaller batch sizes for long sequences
- Clear GPU cache between generations if needed

**Speed Optimization:**
- Use GPU acceleration when available
- Batch multiple prompts together
- Consider model quantization for faster inference

**Quality Control:**
- Set reasonable `max_new_tokens` limits
- Use `eos_token_id` to stop at natural endpoints
- Filter or post-process generated text as needed

### Common Issues and Solutions

**Repetitive Text:**
- Increase temperature (0.8-1.2)
- Use nucleus sampling (top_p=0.9)
- Check for training issues in original model

**Incoherent Text:**
- Decrease temperature (0.5-0.8)
- Use more focused nucleus sampling (top_p=0.7)
- Ensure model was trained adequately

**Short Generations:**
- Check if EOS token is being generated too frequently
- Increase `max_new_tokens`
- Verify `eos_token_id` is set correctly

**Memory Issues:**
- Reduce batch size
- Use gradient checkpointing during training
- Consider model parallelism for very large models

### Integration with Tokenizers

When using with actual tokenizers (e.g., from your BPE implementation):

```python
# Pseudo-code for text-to-text generation
def generate_from_text(model, tokenizer, prompt_text, **kwargs):
    # Encode text to tokens
    prompt_tokens = tokenizer.encode(prompt_text)
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long)
    
    # Generate
    generated = generate_text(model, prompt_tensor, **kwargs)
    
    # Decode back to text
    generated_tokens = generated[0].tolist()
    return tokenizer.decode(generated_tokens)

# Usage
result = generate_from_text(
    model=model,
    tokenizer=your_bpe_tokenizer,
    prompt_text="Once upon a time",
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9
)
print(result)
```

This comprehensive text generation system provides all the tools needed to generate high-quality text from trained Transformer language models with fine-grained control over the sampling process!
