# AdamW Memory and Compute Analysis

This document analyzes the memory and computational requirements for training transformer models with the AdamW optimizer.

## Question (a): Peak Memory Requirements

### Memory Components

#### Parameters Memory
For a transformer model with:
- Vocabulary size: `vocab_size`
- Context length: `context_length` 
- Number of layers: `num_layers`
- Model dimension: `d_model`
- Number of attention heads: `num_heads`
- Feed-forward dimension: `d_ff = 4 × d_model`

**Parameter count per component:**
- Token embeddings: `vocab_size × d_model`
- Position embeddings: `context_length × d_model`
- Per layer:
  - RMSNorm: `d_model` (scale parameters)
  - QKV projections: `3 × d_model × d_model = 3 × d_model²`
  - Attention output projection: `d_model × d_model = d_model²`
  - Feed-forward W1: `d_model × d_ff = d_model × 4d_model = 4 × d_model²`
  - Feed-forward W2: `d_ff × d_model = 4d_model × d_model = 4 × d_model²`
  - RMSNorm: `d_model`
- Final RMSNorm: `d_model`
- Output projection: `d_model × vocab_size`

**Total parameters:**
```
P = vocab_size × d_model + context_length × d_model + 
    num_layers × (2 × d_model + 4 × d_model² + 8 × d_model²) + 
    d_model + d_model × vocab_size

P = 2 × vocab_size × d_model + context_length × d_model + 
    num_layers × (2 × d_model + 12 × d_model²) + d_model

P = d_model × (2 × vocab_size + context_length + 1) + 
    num_layers × d_model × (2 + 12 × d_model)
```

**Parameters memory (float32):** `4P bytes`

#### Activations Memory
For batch size `B = batch_size`:

**Per layer activations:**
- Input/Output: `B × context_length × d_model`
- RMSNorm output: `B × context_length × d_model`
- QKV projections: `3 × B × context_length × d_model`
- Attention scores (Q⊤K): `B × num_heads × context_length × context_length`
- Attention weights (softmax): `B × num_heads × context_length × context_length`
- Attention output: `B × context_length × d_model`
- Feed-forward W1 output: `B × context_length × d_ff = B × context_length × 4 × d_model`
- SiLU output: `B × context_length × 4 × d_model`
- Feed-forward W2 output: `B × context_length × d_model`

**Per layer total:**
```
A_layer = B × context_length × d_model × (1 + 1 + 3 + 1 + 4 + 1) + 
          2 × B × num_heads × context_length²

A_layer = B × context_length × (11 × d_model + 2 × num_heads × context_length)
```

**Final components:**
- Final RMSNorm: `B × context_length × d_model`
- Output embeddings (logits): `B × context_length × vocab_size`
- Cross-entropy (typically negligible): `B × context_length`

**Total activations:**
```
A = num_layers × B × context_length × (11 × d_model + 2 × num_heads × context_length) + 
    B × context_length × (d_model + vocab_size)

A = B × context_length × [num_layers × (11 × d_model + 2 × num_heads × context_length) + 
                          d_model + vocab_size]
```

**Activations memory (float32):** `4A bytes`

#### Gradients Memory
Gradients have the same size as parameters.

**Gradients memory (float32):** `4P bytes`

#### Optimizer State Memory
AdamW maintains two state tensors per parameter:
- `exp_avg` (first moment): same size as parameters
- `exp_avg_sq` (second moment): same size as parameters

**Optimizer state memory (float32):** `8P bytes` (2 × 4P)

### Total Peak Memory

```
Total Memory = Parameters + Activations + Gradients + Optimizer State
Total Memory = 4P + 4A + 4P + 8P = 16P + 4A

Total Memory = 16P + 4 × B × context_length × [num_layers × (11 × d_model + 2 × num_heads × context_length) + d_model + vocab_size]
```

Where:
```
P = d_model × (2 × vocab_size + context_length + 1) + num_layers × d_model × (2 + 12 × d_model)
```

## Question (b): GPT-2 XL Memory Analysis

### GPT-2 XL Parameters
- `vocab_size = 50,257`
- `context_length = 1,024`
- `num_layers = 48`
- `d_model = 1,600`
- `num_heads = 25`
- `d_ff = 4 × 1,600 = 6,400`

### Parameter Count Calculation
```
P = 1,600 × (2 × 50,257 + 1,024 + 1) + 48 × 1,600 × (2 + 12 × 1,600)
P = 1,600 × (100,514 + 1,025) + 48 × 1,600 × (2 + 19,200)
P = 1,600 × 101,539 + 48 × 1,600 × 19,202
P = 162,462,400 + 1,475,993,600
P = 1,638,456,000 ≈ 1.64B parameters
```

### Memory Expression
```
Total Memory = 16P + 4 × batch_size × 1,024 × [48 × (11 × 1,600 + 2 × 25 × 1,024) + 1,600 + 50,257]

Let's calculate the coefficient of batch_size:
- Per layer: 11 × 1,600 + 2 × 25 × 1,024 = 17,600 + 51,200 = 68,800
- All layers: 48 × 68,800 = 3,302,400
- Final terms: 1,600 + 50,257 = 51,857
- Total coefficient: 3,302,400 + 51,857 = 3,354,257

Total Memory = 16 × 1,638,456,000 + 4 × 1,024 × batch_size × 3,354,257
Total Memory = 26,215,296,000 + 13,713,034,752 × batch_size bytes
Total Memory ≈ 26.2GB + 13.7GB × batch_size
```

### Maximum Batch Size for 80GB Memory
```
80GB = 26.2GB + 13.7GB × batch_size
53.8GB = 13.7GB × batch_size
batch_size = 53.8 / 13.7 ≈ 3.93

Maximum batch_size = 3
```

**Answer:** `13.7 × batch_size + 26.2` GB, maximum batch size = 3

## Question (c): AdamW FLOPs per Step

### Forward Pass FLOPs
For each matrix multiplication `A × B` where A is `(m×k)` and B is `(k×n)`:
FLOPs = `2 × m × k × n` (multiply-accumulate operations)

**Per layer FLOPs:**
- QKV projections: `3 × 2 × B × context_length × d_model × d_model`
- Q⊤K computation: `2 × B × num_heads × context_length × d_model/num_heads × context_length`
- Attention weighted sum: `2 × B × num_heads × context_length × context_length × d_model/num_heads`
- Output projection: `2 × B × context_length × d_model × d_model`
- Feed-forward W1: `2 × B × context_length × d_model × 4d_model`
- Feed-forward W2: `2 × B × context_length × 4d_model × d_model`

**Simplifying attention terms:**
- Q⊤K: `2 × B × context_length × d_model × context_length`
- Weighted sum: `2 × B × context_length × context_length × d_model`

**Per layer total:**
```
F_layer = 2 × B × context_length × d_model × (3 × d_model + d_model + 4 × d_model + 4 × d_model) + 
          2 × 2 × B × context_length² × d_model

F_layer = 2 × B × context_length × d_model × (12 × d_model + 2 × context_length)
F_layer = 24 × B × context_length × d_model² + 4 × B × context_length² × d_model
```

**Output embedding:**
`2 × B × context_length × d_model × vocab_size`

**Total forward pass FLOPs:**
```
F_forward = num_layers × (24 × B × context_length × d_model² + 4 × B × context_length² × d_model) + 
            2 × B × context_length × d_model × vocab_size

F_forward = B × context_length × [24 × num_layers × d_model² + 4 × num_layers × context_length × d_model + 2 × d_model × vocab_size]
```

### AdamW Optimizer FLOPs
For each parameter, AdamW performs:
- First moment update: 3 operations (multiply, multiply, add)
- Second moment update: 4 operations (multiply, multiply, multiply, add)  
- Bias correction: 2 exponentiations + 2 operations
- Parameter update: 4 operations (sqrt, add, divide, multiply-subtract)

**Total per parameter:** ~13 operations
**Total optimizer FLOPs:** `13P`

### Total FLOPs per Step
Assuming backward pass = 2 × forward pass:
```
Total FLOPs = 3 × F_forward + 13P
```

## Question (d): Training Time Analysis

### GPT-2 XL Training Setup
- Model: GPT-2 XL (1.64B parameters)
- Training steps: 400,000
- Batch size: 1,024
- Hardware: NVIDIA A100 (19.5 TFLOP/s peak)
- MFU: 50%

### FLOP Calculation for GPT-2 XL
```
B = 1,024
context_length = 1,024
d_model = 1,600
num_layers = 48
vocab_size = 50,257

F_forward = 1,024 × 1,024 × [24 × 48 × 1,600² + 4 × 48 × 1,024 × 1,600 + 2 × 1,600 × 50,257]
F_forward = 1,048,576 × [24 × 48 × 2,560,000 + 4 × 48 × 1,638,400 + 160,822,400]
F_forward = 1,048,576 × [2,949,120,000 + 314,572,800 + 160,822,400]
F_forward = 1,048,576 × 3,424,515,200
F_forward ≈ 3.59 × 10¹⁵ FLOPs per step

Total FLOPs per step = 3 × F_forward = 1.08 × 10¹⁶ FLOPs per step
```

### Training Time Calculation
```
Effective throughput = 19.5 TFLOP/s × 0.5 = 9.75 TFLOP/s = 9.75 × 10¹² FLOP/s

Time per step = (1.08 × 10¹⁶ FLOPs) / (9.75 × 10¹² FLOP/s) = 1,108 seconds

Total training time = 400,000 steps × 1,108 seconds = 443,200,000 seconds

Converting to days:
443,200,000 seconds ÷ (24 × 3600) = 5,130 days ≈ 14 years
```

**Answer:** Approximately **14 years** of training time on a single A100 GPU.

**Note:** This explains why large model training requires:
1. Distributed training across hundreds/thousands of GPUs
2. Model parallelism and data parallelism
3. Significant computational resources and time investment

### Justification
The calculation accounts for:
- Forward pass computational complexity (matrix multiplications dominate)
- Backward pass being twice the forward pass cost
- AdamW optimizer overhead (relatively small compared to forward/backward)
- Hardware efficiency (50% MFU is realistic for large model training)
- The massive scale of modern language model training requirements
