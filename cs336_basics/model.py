

import math
import os
from einops import rearrange, einsum
import einx
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Bool, Int


class Linear(nn.Module):
    def __init__(
            self, in_features: int, out_features: int, 
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
        ):
        """A linear layer initialized with truncated normal fan-in fan-out.

        Args:
            in_features: int
                The number of input features.
            out_features: int
                The number of output features.
            device: Optional[torch.device]
                The device to create the module's parameters on.
            dtype: Optional[torch.dtype]
                The dtype to create the module's parameters with.
        """
        
        super().__init__()
        std = math.sqrt(2 / (in_features + out_features))
        self.weight: Float[Tensor, " out_features in_features"] = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(out_features, in_features, device=device, dtype=dtype), 
                std=std, a=-3*std, b=3*std
            ),
            requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")


class Embedding(nn.Module):
    def __init__(
            self, num_embeddings: int, embedding_dim: int, 
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
        ):
        """A simple embedding layer.

        Args:
            num_embeddings: int
                The number of embeddings. size of the vocabulary
            embedding_dim: int
                The dimension of the embeddings. model dimension
            device: Optional[torch.device]
                The device to create the module's parameters on.
            dtype: Optional[torch.dtype]
                The dtype to create the module's parameters with.
        """
        
        super().__init__()
        std = 1.0
        self.weight: Float[Tensor, " num_embeddings embedding_dim"] = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype), 
                std=std, a=-3 * std, b=3 * std
            ),
            requires_grad=True
        )

    def forward(self, x: Int[Tensor, " ..."]) -> Float[Tensor, " ... embedding_dim"]:
        # Use gather to preserve input dimensions
        # x is a tensor of shape (batch_size, sequence_length)
        return self.weight[x]


class RMSNorm(nn.Module):
    def __init__(
            self, d_model: int, eps: float = 1e-5, 
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
        ):
        """
        RMSNorm is a normalization technique that scales the input by the inverse of the root mean square of the input.
        
        Args:
            d_model: int
                The dimension of the model.
            eps: float
                The epsilon value for numerical stability.
            device: Optional[torch.device]
                The device to create the module's parameters on.
            dtype: Optional[torch.dtype]
                The dtype to create the module's parameters with.
        
        The weight parameter is a learnable scale factor that allows the model to modulate 
        the normalized values independently for each feature dimension, similar to gamma in LayerNorm.
        """
        super().__init__()
        self.weight: Float[Tensor, " d_model"] = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype),
            requires_grad=True
        )
        self.eps = eps

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        # Store original dtype
        orig_dtype = x.dtype
        
        # Upcast to float32 for better numerical precision
        x_f32 = x.to(torch.float32)
        
        # Compute RMSNorm in float32
        rms = torch.rsqrt(torch.mean(x_f32 ** 2, dim=-1, keepdim=True) + self.eps)
        output = x_f32 * rms * self.weight
        
        # Return to original dtype
        return output.to(orig_dtype)

class SwiGLU(nn.Module):
    def __init__(
            self, d_model: int, d_ff: int | None = None,
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
        ):
        """
        SwiGLU is a variant of the GLU activation function that uses SiLU (Swish) activation.
        The feed-forward dimension d_ff is set to approximately 8/3 * d_model, rounded to the nearest multiple of 64.
        
        Args:
            d_model: int
                The input and output dimension
            d_ff: Optional[int]
                The intermediate dimension. If None, will be set to nearest multiple of 64 to 8/3 * d_model
            device: Optional[torch.device]
                The device to create the module's parameters on
            dtype: Optional[torch.dtype]
                The dtype to create the module's parameters with
        """
        super().__init__()
        
        # If d_ff not provided, compute it as nearest multiple of 64 to 8/3 * d_model
        if d_ff is None:
            d_ff = int(round(8/3 * d_model / 64) * 64)
            
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)  # First projection
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)  # Output projection
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)  # Gate projection
        self.act = nn.SiLU()

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        # Project using Linear layers (which use einsum internally)
        w1x = self.w1(x)  # shape: ... d_ff
        w3x = self.w3(x)  # shape: ... d_ff
        
        # Apply SiLU only to w1x and multiply with w3x
        activated = self.act(w1x) * w3x  # shape: ... d_ff
        
        # Project back to d_model dimension
        return self.w2(activated)  # shape: ... d_model
    

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
            self, theta: float, d_k: int, max_seq_len: int, 
            device: torch.device | None = None
        ):
        """
        Rotary Positional Embedding (RoPE) applies position-dependent rotation to the input vectors.
        For a d-dimensional vector at position i, we rotate each pair of elements (2k-1, 2k) by angle θi,k = i * Θ^(2k/d).
        
        Args:
            theta: float
                Base value Θ for computing rotation angles
            d_k: int
                Dimension of query/key vectors (must be even)
            max_seq_len: int
                Maximum sequence length to precompute angles for
            device: Optional[torch.device]
                Device to store the precomputed sin/cos tensors
        """
        super().__init__()
        
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even, got {d_k}")
            
        # Compute frequency bands for each dimension pair
        # For each pair k, the frequency is Θ^(-2k/d)
        dim_indices = torch.arange(0, d_k, 2, device=device).float()
        inv_freq = theta ** (-dim_indices / d_k)  # shape: (d_k//2,)
        
        # For each position i and frequency band k, compute i * Θ^(-2k/d)
        positions = torch.arange(max_seq_len, device=device).float()  # shape: (max_seq_len,)
        angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # shape: (max_seq_len, d_k//2)
        
        # Precompute sin and cos for each angle
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)  # (max_seq_len, d_k//2)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)  # (max_seq_len, d_k//2)
        
        self.d_k = d_k
        self.max_seq_len = max_seq_len

    def forward(self, x: Float[Tensor, " ... seq_len d_k"], token_positions: Int[Tensor, " ... seq_len"]) -> Float[Tensor, " ... seq_len d_k"]:
        """
        Apply rotary position embeddings to the input tensor.
        
        For each position i and dimension pair k:
        [x_{2k}, x_{2k+1}] -> [cos(θ) * x_{2k} - sin(θ) * x_{2k+1}, sin(θ) * x_{2k} + cos(θ) * x_{2k+1}]
        where θ = i * Θ^(-2k/d)
        """
        *batch_dims, seq_len, d_k = x.shape
        assert d_k == self.d_k, f"Input dimension {d_k} doesn't match expected dimension {self.d_k}"
        
        # Get position-specific rotation values
        cos = self.cos_cached[token_positions]  # shape: (..., seq_len, d_k//2)
        sin = self.sin_cached[token_positions]  # shape: (..., seq_len, d_k//2)
        
        # Split input into pairs
        x_even = x[..., ::2]  # shape: (..., seq_len, d_k//2)
        x_odd = x[..., 1::2]  # shape: (..., seq_len, d_k//2)
        
        # Apply rotations using einsum
        # For each pair (x_{2k}, x_{2k+1}):
        # x'_{2k}   = cos(θ) * x_{2k} - sin(θ) * x_{2k+1}
        # x'_{2k+1} = sin(θ) * x_{2k} + cos(θ) * x_{2k+1}
        x_out = torch.empty_like(x)
        x_out[..., ::2] = einsum(cos, x_even, "... seq d_k, ... seq d_k -> ... seq d_k") - \
                          einsum(sin, x_odd, "... seq d_k, ... seq d_k -> ... seq d_k")
        x_out[..., 1::2] = einsum(sin, x_even, "... seq d_k, ... seq d_k -> ... seq d_k") + \
                           einsum(cos, x_odd, "... seq d_k, ... seq d_k -> ... seq d_k")
        
        return x_out
    
def softmax(
        x: Float[Tensor, " ..."], dim: int = -1
    ) -> Float[Tensor, " ..."]:
    """
    Compute softmax along a specified dimension with numerical stability.
    
    Args:
        x: Input tensor of arbitrary shape
        dim: Dimension along which to compute softmax (default: -1, last dimension)
    
    Returns:
        Tensor of same shape as input with softmax applied along specified dimension.
        The values along dim will sum to 1 and form a probability distribution.
    """
    # First compute the maximum value along dim and keep dims for broadcasting
    x_max = torch.max(x, dim=dim, keepdim=True).values
    
    # Subtract max for numerical stability and compute exp
    exp_x = torch.exp(x - x_max)
    
    # Compute sum along dim for normalization
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    
    # Normalize to get probabilities
    return exp_x / sum_exp_x
    
def scaled_dot_product_attention(
        Q: Float[Tensor, "... n_queries d_k"],
        K: Float[Tensor, "... n_keys d_k"],
        V: Float[Tensor, "... n_keys d_v"],
        mask: Float[Tensor, "... n_queries n_keys"] | None = None
    ) -> Float[Tensor, "... n_queries d_v"]:
    """
    Compute scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
    
    Args:
        Q: Query tensor of shape (..., n_queries, d_k)
        K: Key tensor of shape (..., n_keys, d_k)
        V: Value tensor of shape (..., n_keys, d_v)
        mask: Optional mask tensor of shape (..., n_queries, n_keys).
              True values will be included in attention, False values will be masked out.
    
    Returns:
        Output tensor of shape (..., n_queries, d_v)
    """
    # Get the dimension of the key vectors
    d_k = Q.shape[-1]
    
    # Compute attention scores: (Q @ K^T) / sqrt(d_k)
    # Using einsum for clarity and to handle arbitrary batch dimensions
    scores = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        # Set masked positions to -inf before softmax
        # masked_fill approach (efficient) as is a single CUDA kernel
        scores = scores.masked_fill(~mask, float('-inf'))
        # # implement mask fill from scratch. 
        # # Create a tensor of -inf with same shape as scores
        # neg_inf = torch.full_like(scores, float('-inf'))
        # scores = torch.where(mask, scores, neg_inf)
    
    # Apply softmax to get attention weights
    # We want to normalize over the keys dimension
    attn_weights = softmax(scores, dim=-1)  # shape: (..., n_queries, n_keys)
    
    # Apply attention weights to values
    # Again using einsum to handle arbitrary batch dimensions
    return einsum(attn_weights, V, "... q k, ... k d_v -> ... q d_v")


class MultiHeadAttention(nn.Module):
    def __init__(
            self, d_model: int, n_heads: int, 
            max_seq_len: int = 512, rope_theta: float = 10000.0,
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
        ):
        """
        Multi-Head Attention with causal masking and RoPE.
        
        Args:
            d_model: int
                Dimensionality of the model (input and output)
            n_heads: int
                Number of attention heads
            max_seq_len: int
                Maximum sequence length for RoPE precomputation
            rope_theta: float
                Base for RoPE angle calculations
            device: Optional[torch.device]
                Device to create parameters on
            dtype: Optional[torch.dtype]
                Dtype to create parameters with
        """
        super().__init__()
        
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
            
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # dimension of each head's key/query
        self.d_v = d_model // n_heads  # dimension of each head's value
        
        # Create the learnable weight matrices
        # WQ, WK, WV project to (n_heads * d_k) = d_model and will be reshaped later
        self.W_Q = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.n_heads * self.d_k, d_model, device=device, dtype=dtype),
                std=math.sqrt(2 / (2 * d_model)),
                a=-3 * math.sqrt(2 / (2 * d_model)),
                b=3 * math.sqrt(2 / (2 * d_model))
            ),
            requires_grad=True
        )
        self.W_K = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.n_heads * self.d_k, d_model, device=device, dtype=dtype),
                std=math.sqrt(2 / (2 * d_model)),
                a=-3 * math.sqrt(2 / (2 * d_model)),
                b=3 * math.sqrt(2 / (2 * d_model))
            ),
            requires_grad=True
        )
        self.W_V = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.n_heads * self.d_v, d_model, device=device, dtype=dtype),
                std=math.sqrt(2 / (2 * d_model)),
                a=-3 * math.sqrt(2 / (2 * d_model)),
                b=3 * math.sqrt(2 / (2 * d_model))
            ),
            requires_grad=True
        )
        
        # WO projects back to d_model
        self.W_O = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.n_heads * self.d_v, d_model, device=device, dtype=dtype),
                std=math.sqrt(2 / (2 * d_model)),
                a=-3 * math.sqrt(2 / (2 * d_model)),
                b=3 * math.sqrt(2 / (2 * d_model))
            ),
            requires_grad=True
        )
        
        # Initialize RoPE for query and key rotations
        self.rope = RotaryPositionalEmbedding(rope_theta, self.d_k, max_seq_len, device)

    def forward(
            self, x: Float[Tensor, "... seq_len d_model"], 
            token_positions: Int[Tensor, "... seq_len"] | None = None,
        ) -> Float[Tensor, "... seq_len d_model"]:
        """
        Apply causal multi-head self-attention to the input.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Positions for RoPE, shape (..., seq_len)
            
        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        batch_dims = x.shape[:-2]
        seq_len = x.shape[-2]
        
        # Project and reshape to separate heads
        # First project, then reshape to separate heads
        Q = einsum(x, self.W_Q, "... seq d_in, d_k d_in -> ... seq d_k")
        Q = rearrange(Q, "b seq_len (heads d_k) -> b heads seq_len d_k", heads=self.n_heads)
        
        K = einsum(x, self.W_K, "... seq d_in, d_k d_in -> ... seq d_k")
        K = rearrange(K, "b seq_len (heads d_k) -> b heads seq_len d_k", heads=self.n_heads)
        
        V = einsum(x, self.W_V, "... seq d_in, d_v d_in -> ... seq d_v")
        V = rearrange(V, "b seq_len (heads d_v) -> b heads seq_len d_v", heads=self.n_heads)
        
        # Apply RoPE to Q and K (treating head dim as batch dim)
        if token_positions is not None:
            Q_flat = rearrange(Q, "b heads seq d_k -> (b heads) seq d_k")
            K_flat = rearrange(K, "b heads seq d_k -> (b heads) seq d_k")
            
            # Expand token_positions for each head
            pos_expanded = token_positions.unsqueeze(-2).expand(*batch_dims, self.n_heads, seq_len).reshape(-1, seq_len)
            
            # Apply RoPE
            Q = rearrange(self.rope(Q_flat, pos_expanded), "(b heads) seq d_k -> b heads seq d_k", heads=self.n_heads)
            K = rearrange(self.rope(K_flat, pos_expanded), "(b heads) seq d_k -> b heads seq d_k", heads=self.n_heads)
            
        # Create causal mask
        # shape: (seq_len, seq_len), True where attention is allowed
        # triu(k=1) gets upper triangular part excluding diagonal
        # ~triu(k=1) gets lower triangular part including diagonal
        causal_mask = ~torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        
        # Apply scaled dot-product attention with causal mask
        # Treat batch and head dims as batch dims
        attn_flat = scaled_dot_product_attention(
            rearrange(Q, "b heads seq d_k -> (b heads) seq d_k"),
            rearrange(K, "b heads seq d_k -> (b heads) seq d_k"),
            rearrange(V, "b heads seq d_v -> (b heads) seq d_v"),
            causal_mask
        )
        
        # Reshape attention output back to separate heads
        attn = rearrange(attn_flat, "(b heads) seq d_v -> b heads seq d_v", heads=self.n_heads)
        
        # Concatenate heads and project back to d_model
        out = rearrange(attn, "b heads seq d_v -> b seq (heads d_v)")
        return einsum(out, self.W_O, "... seq d_v, d_model d_v -> ... seq d_model")
    

class TransformerBlock(nn.Module):
    def __init__(
            self, d_model: int, num_heads: int, d_ff: int,
            max_seq_len: int = 512, rope_theta: float = 10000.0,
            device: torch.device | None = None, 
            dtype: torch.dtype | None = None
        ):
        """
        Pre-norm Transformer block with RMSNorm, MultiHeadAttention, and SwiGLU feed-forward network.
        
        Args:
            d_model: int
                Dimensionality of the model (input and output)
            num_heads: int
                Number of attention heads
            d_ff: int
                Dimensionality of the feed-forward inner layer
            max_seq_len: int
                Maximum sequence length for RoPE precomputation
            rope_theta: float
                Base for RoPE angle calculations
            device: Optional[torch.device]
                Device to create parameters on
            dtype: Optional[torch.dtype]
                Dtype to create parameters with
        """
        super().__init__()
        
        # First sublayer: Multi-head attention with RMSNorm
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadAttention(d_model, num_heads, max_seq_len, rope_theta, device, dtype)
        
        # Second sublayer: Feed-forward with RMSNorm
        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ff = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
            self, x: Float[Tensor, "... seq_len d_model"],
            token_positions: Int[Tensor, "... seq_len"] | None = None,
        ) -> Float[Tensor, "... seq_len d_model"]:
        """
        Apply the Transformer block to the input.
        
        Args:
            x: Input tensor of shape (..., seq_len, d_model)
            token_positions: Optional positions for RoPE, shape (..., seq_len)
            
        Returns:
            Output tensor of shape (..., seq_len, d_model)
        """
        if token_positions is None:
            seq_len = x.shape[-2]
            token_positions = torch.arange(seq_len, device=x.device)
            token_positions = token_positions.expand(x.shape[:-1])

        # First sublayer: Multi-head attention with pre-norm and residual
        # y = x + MultiHeadSelfAttention(RMSNorm(x))
        attn_out = self.attn(self.norm1(x), token_positions)
        x = x + attn_out
        
        # Second sublayer: Feed-forward with pre-norm and residual
        # y = x + FeedForward(RMSNorm(x))
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        
        return x


class TransformerLM(nn.Module):
    def __init__(
            self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
        ):
        """
        A complete Transformer-based language model.

        Args:
            vocab_size: The number of unique tokens in the vocabulary.
            context_length: The maximum sequence length the model can handle.
            d_model: The dimensionality of the model's embeddings and hidden states.
            num_layers: The number of TransformerBlocks to stack.
            num_heads: The number of attention heads in each TransformerBlock.
            d_ff: The dimensionality of the feed-forward network's inner layer.
            rope_theta: The base value for RoPE frequency calculations.
            device: The device to create the module's parameters on.
            dtype: The dtype to create the module's parameters with.
        """
        super().__init__()
        
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    rope_theta=rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        
        self.norm_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(
            self, x: Int[Tensor, "batch_size seq_len"],
        ) -> Float[Tensor, "batch_size seq_len vocab_size"]:
        """
        Forward pass for the Transformer Language Model.

        Args:
            x: Input tensor of token indices.

        Returns:
            Logits over the vocabulary for each token in the sequence.
        """
        batch_size, seq_len = x.shape
        
        # 1. Get token embeddings
        h = self.token_embeddings(x)
        
        # 2. Create token positions for RoPE
        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # 3. Pass through Transformer blocks
        for layer in self.layers:
            h = layer(h, token_positions=token_positions)
        
        # 4. Final normalization
        h = self.norm_final(h)
        
        # 5. Language model head
        logits = self.lm_head(h)
        
        return logits