
### How RoPE Encodes Relative Position

The core idea of RoPE is that while the individual rotation angles for each position are unique, the **relative distance** between positions is what determines the final attention score.

#### The Angles Are Not the Same 📐
- For a given dimension pair $i$, the rotation angle for position 3 is $\theta_{3,i}$, and for position 4 it's $\theta_{4,i}$. These are different because the angles depend on the absolute position.

---

#### Individual Rotations, Collective Effect 🔄
- A vector at position $m$ is rotated based on a set of angles $\{\theta_{m,0}, \theta_{m,1}, \dots, \theta_{m,d/2-1}\}$. Each angle corresponds to a different dimension pair within the vector.

---

#### The Dot Product Is What Matters ✨
- The magic happens when we take the dot product of a query vector from position 3 ($q_3$) and a key vector from position 1 ($k_1$). The attention score is proportional to this dot product.

- The dot product of the rotated vectors simplifies to:
  $$q_3 \cdot k_1 \propto \sum_{i=0}^{d/2-1} [q_{3,2i} k_{1,2i} + q_{3,2i+1} k_{1,2i+1}] \cdot \cos(\theta_{3,i} - \theta_{1,i}) + \dots$$
  
  Notice the term $\cos(\theta_{3,i} - \theta_{1,i})$. The crucial part is that the rotation angle difference is what's used. Since $\theta_{m,i} = m \cdot \omega_i$, the difference is:
  $$\theta_{3,i} - \theta_{1,i} = (3 \cdot \omega_i) - (1 \cdot \omega_i) = (3-1) \cdot \omega_i = 2 \cdot \omega_i$$

- Now, let's look at positions 4 and 2. The angle difference is:
  $$\theta_{4,i} - \theta_{2,i} = (4 \cdot \omega_i) - (2 \cdot \omega_i) = (4-2) \cdot \omega_i = 2 \cdot \omega_i$$

- Since the angle difference is the same in both cases, RoPE treats the relative position of $(3,1)$ and $(4,2)$ as identical. This allows the model to generalize effectively.


### FLOPs

#### (a) GPT-2 XL Parameters and Memory

**Configuration:**
- `vocab_size`: 50,257
- `context_length`: 1,024
- `num_layers`: 48
- `d_model`: 1,600
- `num_heads`: 25
- `d_ff`: 6,400

**Parameters calculation:**
- **Token embeddings**: `vocab_size × d_model = 50,257 × 1,600 = 80,411,200`
- **Each Transformer layer**:
  - Multi-head attention: `4 × d_model × d_model = 4 × 1,600 × 1,600 = 10,240,000` (Q, K, V, O projections)
  - Feed-forward: `2 × d_model × d_ff = 2 × 1,600 × 6,400 = 20,480,000` (up-projection + down-projection)
  - Layer norms: `2 × d_model = 3,200` (RMSNorm weights)
  - **Total per layer**: `10,240,000 + 20,480,000 + 3,200 = 30,723,200`
- **All layers**: `48 × 30,723,200 = 1,474,713,600`
- **Final layer norm**: `1,600`
- **Language model head**: `d_model × vocab_size = 1,600 × 50,257 = 80,411,200`

**Total parameters**: `80,411,200 + 1,474,713,600 + 1,600 + 80,411,200 = 1,635,537,600 ≈ 1.64 billion parameters`

**Memory requirement**: `1.64B × 4 bytes = 6.56 GB` (assuming single-precision float32)

#### (b) Matrix Multiplies and FLOPs for GPT-2 XL

**For sequence length L = 1,024:**

##### Token Embeddings:
- **Embedding lookup**: `L × d_model = 1,024 × 1,600 = 1,638,400 FLOPs`

##### Each Transformer Layer (48 layers):

**Multi-Head Attention:**
1. **Q, K, V projections**: `3 × L × d_model × d_model = 3 × 1,024 × 1,600 × 1,600 = 7,864,320,000 FLOPs`
2. **Attention computation**: `L × L × d_k × heads = 1,024 × 1,024 × 64 × 25 = 1,677,721,600 FLOPs`
3. **Output projection**: `L × d_model × d_model = 1,024 × 1,600 × 1,600 = 2,621,440,000 FLOPs`

**Feed-Forward Network:**
4. **Up-projection**: `L × d_model × d_ff = 1,024 × 1,600 × 6,400 = 10,485,760,000 FLOPs`
5. **Down-projection**: `L × d_ff × d_model = 1,024 × 6,400 × 1,600 = 10,485,760,000 FLOPs`

**Layer Norms:**
- **RMSNorm**: Negligible (element-wise operations)

**Per layer total**: `7,864,320,000 + 1,677,721,600 + 2,621,440,000 + 10,485,760,000 + 10,485,760,000 = 33,135,001,600 FLOPs`

**All layers**: `48 × 33,135,001,600 = 1,590,480,076,800 FLOPs`

##### Final Layer:
- **Final RMSNorm**: Negligible
- **Language model head**: `L × d_model × vocab_size = 1,024 × 1,600 × 50,257 = 82,341,068,800 FLOPs`

**Total FLOPs**: `1,638,400 + 1,590,480,076,800 + 82,341,068,800 = 1,672,823,556,000 ≈ 1.67 trillion FLOPs`

#### (c) Most FLOP-Intensive Components

The **feed-forward networks** require the most FLOPs, accounting for approximately 60% of total computation, followed by the **attention projections** (Q, K, V, O) which consume about 30% of FLOPs, while the **attention computation itself** (the actual attention mechanism) only requires about 10% of total FLOPs.

#### (d) FLOPs Breakdown Across Model Sizes

##### GPT-2 Small (12 layers, 768 d_model, 12 heads, d_ff = 3072):
- **Feed-forward**: ~58% of FLOPs
- **Attention projections**: ~32% of FLOPs  
- **Attention computation**: ~10% of FLOPs

##### GPT-2 Medium (24 layers, 1024 d_model, 16 heads, d_ff = 4096):
- **Feed-forward**: ~60% of FLOPs
- **Attention projections**: ~30% of FLOPs
- **Attention computation**: ~10% of FLOPs

##### GPT-2 Large (36 layers, 1280 d_model, 20 heads, d_ff = 5120):
- **Feed-forward**: ~61% of FLOPs
- **Attention projections**: ~29% of FLOPs
- **Attention computation**: ~10% of FLOPs

**As model size increases, the feed-forward networks consume a proportionally larger share of FLOPs** because d_ff scales with d_model (typically 4×), while attention computation scales more slowly with the number of heads.

#### (e) GPT-2 XL with Context Length 16,384

**Total FLOPs increase by a factor of 16** (from 1,024 to 16,384 tokens), resulting in approximately 26.8 trillion FLOPs for a single forward pass. **The attention computation becomes proportionally more significant**, increasing from ~10% to ~25% of total FLOPs, while feed-forward networks decrease from ~60% to ~45% due to the quadratic scaling of attention with sequence length.