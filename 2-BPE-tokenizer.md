# BPE Tokenizer Implementation Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture and Design](#architecture-and-design)
3. [BPE Training Implementation](#bpe-training-implementation)
4. [Tokenizer Implementation](#tokenizer-implementation)
5. [Performance Optimization](#performance-optimization)
6. [Usage Examples](#usage-examples)
7. [Configuration and Tuning](#configuration-and-tuning)

## Overview

This document describes the implementation of a complete BPE (Byte Pair Encoding) tokenizer system with automatic strategy selection, parallel processing, and memory-efficient handling of large corpora. The system consists of two main components:

- **BPE Training** (`train_bpe.py`): Efficient training on corpora of any size
- **Tokenizer** (`tokenizer.py`): Smart encoding with automatic strategy selection

### Key Features

- **Automatic Strategy Selection**: Chooses optimal processing approach based on data size
- **Parallel Processing**: Multi-core processing for large files
- **Memory Efficiency**: Streaming and iterator-based approaches for huge corpora
- **Special Token Handling**: Proper preservation of special tokens during training and encoding
- **Configurable Thresholds**: Easy tuning for different use cases

## Architecture and Design

### Configuration-Driven Design

Both components use centralized configuration classes that define size thresholds and processing strategies:

```python
# BPE Training Configuration
class BPEConfig:
    SMALL_FILE_THRESHOLD = 1MB      # Sequential processing
    MEDIUM_FILE_THRESHOLD = 1GB     # Parallel list processing  
    LARGE_FILE_THRESHOLD = 5GB      # Parallel iterator processing
    MEMORY_EFFICIENT_THRESHOLD = 5GB # Streaming trainer

# Tokenizer Configuration  
class TokenizerConfig:
    SMALL_TEXT_THRESHOLD = 10K      # Simple encoding
    MEDIUM_TEXT_THRESHOLD = 1M      # Iterable encoding
    LARGE_TEXT_THRESHOLD = 10M      # Memory-efficient encoding
    SMALL_FILE_THRESHOLD = 1MB      # Simple file reading
    LARGE_FILE_THRESHOLD = 100MB    # Parallel processing
```

### Strategy Enumeration

The system defines clear strategies for different performance profiles:

```python
class TokenizationStrategy:
    SEQUENTIAL = "sequential"           # Single-threaded, lowest memory
    PARALLEL_LIST = "parallel_list"     # Multi-threaded with lists, fastest
    PARALLEL_ITER = "parallel_iter"     # Multi-threaded with iterators, balanced
    PARALLEL_STREAMING = "parallel_streaming"  # Multi-threaded streaming, lowest memory
```

## BPE Training Implementation

### Three-Tier Processing Strategy

The BPE training system automatically chooses the optimal approach based on corpus size:

| Corpus Size | Strategy | Memory Usage | Speed | Best For |
|-------------|----------|--------------|-------|----------|
| < 1MB | Sequential processing | Minimal | Fastest | Small datasets |
| 1MB - 1GB | Parallel + Lists | Moderate | Fast | Medium datasets |
| 1GB - 5GB | Parallel + Iterators | Low | Moderate | Large datasets |
| > 5GB | Parallel + Streaming | Minimal | Moderate | Huge datasets |

### Core Components

#### 1. Chunk Boundary Detection

```python
def find_chunk_boundaries(file, desired_num_chunks, split_special_tokens):
    """
    Finds optimal chunk boundaries that don't split special tokens.
    
    Key Features:
    - Respects special token boundaries
    - Handles multiple special tokens
    - Uses mini-chunks for efficient boundary finding
    - Returns unique, sorted boundaries
    """
```

**Why Not Parallelize Boundary Finding?**
- **I/O Bound**: Limited by disk I/O, not CPU
- **Sequential Access**: Optimal for disk performance
- **Coordination Overhead**: Parallel approach would be complex
- **Fast Enough**: Boundary finding is already efficient

#### 2. Tokenization Functions

The system provides multiple tokenization approaches:

```python
def process_chunk_basic(chunk_text, special_tokens) -> List[bytes]:
    """Basic chunk processing - returns list of tokens."""
    
def process_chunk_iter(chunk_text, special_tokens) -> Iterator[bytes]:
    """Iterator chunk processing - yields tokens one by one."""
    
def process_chunk_bytes(chunk, special_tokens) -> List[bytes]:
    """Process bytes chunk - bridges byte and string processing."""
```

#### 3. Parallel Processing Implementations

```python
def tokenize_sequential(input_path, special_tokens) -> Iterator[bytes]:
    """Single-threaded processing for small files."""
    
def tokenize_parallel_list(input_path, special_tokens) -> List[bytes]:
    """Parallel processing with lists - fastest but highest memory."""
    
def tokenize_parallel_iter(input_path, special_tokens) -> Iterator[bytes]:
    """Parallel processing with iterators - balanced approach."""
    
def tokenize_parallel_streaming(input_path, special_tokens) -> Iterator[bytes]:
    """Ultra memory-efficient parallel processing."""
```

#### 4. BPE Trainer Classes

**Standard BPETrainer**
```python
class BPETrainer:
    """
    In-memory BPE trainer for small to medium corpora.
    
    Key Data Structures:
    - vocab: Dict[int, bytes] - Maps token IDs to byte sequences
    - words: List[List[int]] - List of words, each word is a list of token IDs
    - pair_counts: Counter - Counts frequency of each token pair
    - word_pair_indices: Dict[int, List[Tuple[int, Tuple[int, int]]]] 
      - Maps word_idx → [(position, pair), ...]
    - pair_to_words: Dict[Tuple[int, int], Set[int]]
      - Maps pair → set of word indices containing this pair
    """
```

**Memory-Efficient BPETrainer**
```python
class BPETrainerMemoryEfficient:
    """
    Memory-efficient BPE trainer for very large corpora.
    
    Instead of storing all words in memory, this trainer:
    1. Builds initial pair counts by streaming through words once
    2. Re-processes the corpus for each merge step
    3. Uses significantly less memory at the cost of more I/O
    """
```

### The Merge Process: Step-by-Step

#### Finding the Best Pair

```python
def get_best_pair(self) -> Tuple[int, int]:
    """Find the most frequent pair with lexicographical tie-breaking."""
    best_pair = max(
        self.pair_counts.keys(),
        key=lambda p: (
            self.pair_counts[p],           # Primary: frequency (higher is better)
            self.vocab.get(p[0], b""),     # Secondary: first token bytes (lexicographically larger)
            self.vocab.get(p[1], b""),     # Tertiary: second token bytes (lexicographically larger)
        ),
    )
    return best_pair
```

#### Executing the Merge

```python
def merge_pair(self, pair: Tuple[int, int]) -> None:
    """
    Merge a specific pair and update all data structures.
    
    Algorithm:
    1. Create new token by concatenating the two tokens
    2. Find all words that contain this pair
    3. Process each affected word:
       a. Remove old pair statistics for this word
       b. Apply the merge to create new word
       c. Update the word and rebuild pair statistics
    """
```

**Example Merge Process:**
```
Before merge (pair = (108, 108), vocab[108] = b'l'):
    word = [72, 101, 108, 108, 111]  # "Hello"
    pairs = [(72,101), (101,108), (108,108), (108,111)]
    
After merge (new_token_id = 256, vocab[256] = b'll'):
    word = [72, 101, 256, 111]      # "He" + "ll" + "o"
    pairs = [(72,101), (101,256), (256,111)]
```

### Main Training Interface

```python
def run_train_bpe(input_path, vocab_size, special_tokens, strategy="auto"):
    """
    Main interface for BPE training with automatic optimization.
    
    Features:
    - Automatic strategy selection based on file size
    - Memory-efficient trainer for very large files
    - Parallel processing for medium to large files
    - Progress feedback and strategy reporting
    """
```

## Tokenizer Implementation

### Automatic Strategy Selection

The tokenizer automatically chooses the best encoding strategy:

```python
def encode_auto(self, text: str, strategy: str = "auto", verbose: bool = False) -> List[int]:
    """
    Automatically choose the best encoding strategy based on text size.
    
    Strategies:
    - "simple": Always use simple encode() method  
    - "iterable": Use streaming encode_iterable() method
    - "large_memory": Use memory-efficient approach for very large texts
    """
```

| Text Size | Strategy | Memory Usage | Speed | Best For |
|-----------|----------|--------------|-------|----------|
| < 10K | simple | Low | Fast | Real-time inference |
| 10K-1M | iterable | Medium | Medium | Document processing |
| > 1M | large_memory | Low | Fast | Batch preprocessing |

### File Encoding with Automatic Selection

```python
def encode_file_auto(self, file_path, strategy: str = "auto", verbose: bool = False):
    """
    Automatically choose the best file encoding strategy based on file size.
    
    Strategies:
    - "simple_file": Read entire file and use simple encode()
    - "large_file": Use parallel processing for large files
    """
```

| File Size | Strategy | Cores | Memory | Speed | Best For |
|-----------|----------|-------|--------|-------|----------|
| < 1MB | simple_file | 1 | Medium | Fast | Small documents |
| > 100MB | large_file | 8 | Low | V.Fast | Training corpus |

### Core Encoding Methods

#### 1. Simple Encoding (`_encode_simple`)

```python
def _encode_simple(self, text: str) -> List[int]:
    """
    Encode text using the simple encode() method.
    
    Two-phase process:
    1. Handle special tokens first (they have priority)
    2. Process remaining text with BPE tokenization
    """
```

**Example Process:**
```
Input: "Hello<|endoftext|>World"
Special tokens: ["<|endoftext|>"]

Phase 1: "Hello" - no special token at start → process as regular text
Phase 2: "<|endoftext|>" - found special token → encode as single token  
Phase 3: "World" - no special token at start → process as regular text
```

#### 2. Iterable Encoding (`encode_iterable`)

```python
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    """
    Streaming tokenization with token boundary preservation.
    
    Algorithm:
    1. Accumulate text chunks in a buffer
    2. Find complete token boundaries using GPT-2 regex
    3. Process only complete tokens, keep incomplete ones for next iteration
    4. Yield token IDs one by one
    """
```

**Example Streaming Process:**
```
chunks = ["Hello, wor", "ld! How are", " you?"]

Iteration 1: buffer = "Hello, wor"
    Complete tokens: ["Hello", ","] (ends at position 6)
    Remaining: " wor"
    Yields: [token_ids for "Hello", ","]
    
Iteration 2: buffer = " world! How are" 
    Complete tokens: [" wor", "ld", "!", " How"] (ends at position 11)
    Remaining: " are"
    Yields: [token_ids for " wor", "ld", "!", " How"]
    
Iteration 3: buffer = " are you?"
    Final processing: yields all remaining tokens
```

#### 3. Large File Encoding (`encode_large_file`)

```python
def encode_large_file(self, file_path, chunk_size_mb: int = 100):
    """
    Encode very large files efficiently using parallel processing.
    
    Features:
    - Parallel processing: uses multiple CPU cores
    - Memory efficient: processes file in chunks
    - Special token aware: respects special token boundaries
    - Streaming output: yields token IDs as computed
    - Automatic strategy: chooses best approach based on file size
    """
```

### BPE Merge Application

```python
def _apply_merges_to_token(self, token: bytes) -> List[bytes]:
    """
    Apply BPE merges to a single token using learned merge rules.
    
    Algorithm:
    1. Start with individual bytes as the initial segmentation
    2. Repeatedly find the highest-priority merge (lowest rank number)
    3. Apply the merge to combine adjacent segments
    4. Continue until no more merges are possible
    """
```

**Example Merge Process:**
```
token = b'hello'
merge_ranks = {(b'h', b'e'): 0, (b'l', b'l'): 1, (b'he', b'llo'): 2}

Step 1: parts = [b'h', b'e', b'l', b'l', b'o']
Step 2: Find best merge - (b'h', b'e') has rank 0 (highest priority)
Step 3: parts = [b'he', b'l', b'l', b'o']
Step 4: Find best merge - (b'l', b'l') has rank 1
Step 5: parts = [b'he', b'll', b'o']  
Step 6: Find best merge - (b'he', b'll') not in ranks, no merge for (b'll', b'o')
Result: [b'he', b'll', b'o']
```

## Performance Optimization

### Memory Efficiency Strategies

#### 1. Iterator-Based Processing

```python
# Traditional approach (memory-intensive)
pre_tokens = parallel_pre_tokenize(file)  # Loads all tokens into memory
word_list = [convert(token) for token in pre_tokens]  # Another full copy
trainer = BPETrainer(vocab, word_list)  # Yet another copy

# Memory-efficient approach (iterator-based)
token_iter = parallel_pre_tokenize_iter(file)  # Yields tokens one by one
word_iter = (convert(token) for token in token_iter)  # Generator chain
trainer = BPETrainerMemoryEfficient(vocab, word_iter)  # Streams through corpus
```

#### 2. Streaming BPE Training

The `BPETrainerMemoryEfficient` class:
- Doesn't store words in memory - re-reads corpus for each merge
- Streams through data - processes one word at a time
- Trades I/O for memory - suitable for corpora that don't fit in RAM

#### 3. Chunked File Processing

```python
def find_chunk_boundaries(file, desired_num_chunks, split_special_tokens):
    """
    Efficient chunking that respects special token boundaries.
    
    Uses mini-chunks (4KB) for boundary detection to minimize memory usage
    while ensuring accurate boundary placement.
    """
```

### Parallel Processing Optimization

#### 1. Multiprocessing Strategies

```python
# For medium files: parallel list processing
with mp.Pool(n_cores) as pool:
    results = pool.starmap(process_chunk_basic, text_chunks)

# For large files: parallel iterator processing  
with mp.Pool(n_cores) as pool:
    for chunk_results in pool.imap(process_chunk_iter, chunks):
        yield from chunk_results

# For huge files: parallel streaming processing
with ProcessPoolExecutor(max_workers=n_cores) as executor:
    for future in as_completed(future_to_chunk):
        yield from future.result()
```

#### 2. CPU Core Utilization

```python
n_cores = max(1, mp.cpu_count() - 1)  # Leave one core free for system
```

### Real-World Performance

For the 11GB OpenWebText corpus:

| Approach | Peak Memory | Time | Notes |
|----------|-------------|------|-------|
| Original | ~40GB | 30 min | May cause OOM |
| Iterator | ~4GB | 45 min | Always works |
| Memory Trainer | ~2GB | 60 min | Works on any machine |

## Usage Examples

### 1. Real-time Inference

```python
# Use default encode() - automatically selects best strategy
tokens = tokenizer.encode(user_input)
```

### 2. Document Processing

```python
# Process multiple documents efficiently
for doc in documents:
    tokens = tokenizer.encode_auto(doc, strategy="auto", verbose=True)
```

### 3. Large Dataset Preprocessing

```python
# Process huge training files
for token_id in tokenizer.encode_file_auto("huge_corpus.txt"):
    training_batch.append(token_id)
```

### 4. Memory-Constrained Environments

```python
# Force memory-efficient strategies
tokens = tokenizer.encode_auto(text, strategy="iterable") 
```

### 5. BPE Training

```python
# Automatic selection (recommended)
vocab, merges = run_train_bpe("huge_corpus.txt", 50000, ["<|endoftext|>"])

# Force memory-efficient mode
vocab, merges = run_train_bpe("any_corpus.txt", 50000, ["<|endoftext|>"], 
                             force_memory_efficient=True)

# Manual iterator usage
token_iter = parallel_pre_tokenize_iter("huge_corpus.txt", ["<|endoftext|>"])
for token in token_iter:
    process(token)  # Process one token at a time
```

### 6. Custom Thresholds

```python
# Adjust thresholds for your use case
TokenizerConfig.SMALL_TEXT_THRESHOLD = 5_000  # More aggressive
TokenizerConfig.LARGE_FILE_THRESHOLD = 50 * 1024 * 1024  # 50MB

BPEConfig.SMALL_FILE_THRESHOLD = 512 * 1024  # 512KB
BPEConfig.MEMORY_EFFICIENT_THRESHOLD = 2 * 1024 * 1024 * 1024  # 2GB
```

## Configuration and Tuning

### Tokenizer Configuration

```python
class TokenizerConfig:
    # Text size thresholds (in characters)
    SMALL_TEXT_THRESHOLD = 10_000        # 10K chars - use simple encode()
    MEDIUM_TEXT_THRESHOLD = 1_000_000    # 1M chars - use encode_iterable() 
    LARGE_TEXT_THRESHOLD = 10_000_000    # 10M chars - consider encode_large_file()
    
    # File size thresholds (in bytes) 
    SMALL_FILE_THRESHOLD = 1024 * 1024           # 1MB - use simple file reading
    LARGE_FILE_THRESHOLD = 100 * 1024 * 1024     # 100MB - use parallel processing
    
    # Memory and performance settings
    CHUNK_SIZE_MB = 100                  # Default chunk size for file processing
    BUFFER_SIZE_CHARS = 50_000          # Buffer size for streaming
```

### BPE Training Configuration

```python
class BPEConfig:
    # File size thresholds (in bytes)
    SMALL_FILE_THRESHOLD = 1024 * 1024          # 1MB - use sequential processing
    MEDIUM_FILE_THRESHOLD = 1024 * 1024 * 1024  # 1GB - use parallel list processing
    LARGE_FILE_THRESHOLD = 5 * 1024 * 1024 * 1024  # 5GB - use parallel iterator
    MEMORY_EFFICIENT_THRESHOLD = 5 * 1024 * 1024 * 1024  # 5GB - use streaming trainer
    
    # Chunk processing thresholds
    LARGE_CHUNK_THRESHOLD = 100 * 1024 * 1024   # 100MB - use memory-efficient chunk processing
    LARGE_TEXT_THRESHOLD = 1024 * 1024          # 1MB - use finditer instead of findall
    
    # Processing settings
    MINI_CHUNK_SIZE = 4096                      # 4KB - for boundary finding
    MULTIPROCESSING_CHUNKSIZE = 1               # Chunk size for pool.imap
```

### Performance Tuning Guidelines

#### For High-Memory Systems
- Lower thresholds to use faster strategies more often
- Increase chunk sizes for better parallel efficiency
- Use `parallel_list` strategy for medium files

#### For Memory-Constrained Systems
- Increase thresholds to use memory-efficient strategies
- Decrease chunk sizes to reduce peak memory usage
- Use `parallel_streaming` strategy for large files

#### For High-CPU Systems
- Increase number of worker processes
- Use `parallel_iter` strategy for better CPU utilization
- Adjust `MULTIPROCESSING_CHUNKSIZE` for optimal load balancing

### Monitoring and Debugging

#### Verbose Mode
```python
# Enable verbose output to see strategy selection
tokens = tokenizer.encode_auto(text, verbose=True)
# Output: "Encoding 50K chars using 'iterable' strategy"

vocab, merges = run_train_bpe("corpus.txt", 50000, ["<|endoftext|>"], verbose=True)
# Output: "Using parallel_iter strategy for 2.5GB corpus"
```

#### Memory Monitoring
```python
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Monitor memory during processing
print(f"Memory before: {monitor_memory():.1f}MB")
tokens = list(tokenizer.encode_file_auto("large_file.txt"))
print(f"Memory after: {monitor_memory():.1f}MB")
```

## Key Innovations

### 1. Automatic Strategy Selection
- No manual tuning required
- Optimal performance for any input size
- Clear strategy reporting for transparency

### 2. Memory-Efficient Processing
- `finditer()` instead of `findall()` - Processes regex matches one by one
- Generator chains - Avoid intermediate lists at every step
- Streaming BPE trainer - Re-reads corpus instead of storing in memory

### 3. Parallel Processing Optimization
- Multiple parallel strategies for different use cases
- Efficient chunk boundary detection
- Load balancing with configurable chunk sizes

### 4. Special Token Handling
- Boundary-aware chunking
- Priority processing during encoding
- Consistent handling across all strategies

### 5. Configuration-Driven Design
- Centralized thresholds for easy tuning
- Clear separation of concerns
- Extensible architecture for new strategies

This system can now handle the largest corpora (like the 11GB OpenWebText) on machines with limited RAM, making BPE training and tokenization accessible for very large-scale language model training!
