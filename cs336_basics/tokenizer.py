import json
from typing import Dict, List, Tuple, Iterable, Iterator, Optional
from pathlib import Path
import os

from .train_bpe import process_chunk_basic, PAT


# ============================================================================
# TOKENIZER CONFIGURATION
# ============================================================================

class TokenizerConfig:
    """Configuration for tokenizer encoding strategies with automatic selection."""
    
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
    
    @classmethod
    def get_text_strategy(cls, text_length: int) -> str:
        """Determine the optimal encoding strategy based on text length."""
        if text_length < cls.SMALL_TEXT_THRESHOLD:
            return "simple"
        elif text_length < cls.MEDIUM_TEXT_THRESHOLD:
            return "iterable"  
        else:
            return "large_memory"
    
    @classmethod
    def get_file_strategy(cls, file_size: int) -> str:
        """Determine the optimal encoding strategy based on file size."""
        if file_size < cls.SMALL_FILE_THRESHOLD:
            return "simple_file"
        else:
            return "large_file"
    
    @classmethod
    def format_size(cls, size: int, unit: str = "chars") -> str:
        """Format size in human-readable format."""
        if unit == "chars":
            if size < 1_000:
                return f"{size} chars"
            elif size < 1_000_000:
                return f"{size // 1_000}K chars"
            else:
                return f"{size // 1_000_000}M chars"
        else:  # bytes
            if size < 1024 * 1024:
                return f"{size // 1024}KB"
            elif size < 1024 * 1024 * 1024:
                return f"{size // (1024 * 1024)}MB"
            else:
                return f"{size // (1024 * 1024 * 1024)}GB"


# ============================================================================
# MAIN TOKENIZER CLASS
# ============================================================================

class BPETokenizer:
    """BPE Tokenizer with automatic encoding strategy selection."""
    
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ):
        """Initialize the tokenizer with vocabulary and merges.
        
        Args:
            vocab: Dictionary mapping token IDs to byte sequences
            merges: List of merge rules as (bytes, bytes) tuples
            special_tokens: Optional list of special tokens to add to vocabulary
        """
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # Create reverse vocab for encoding
        self.byte_to_id = {v: k for k, v in vocab.items()}
        
        # Add special tokens if provided and not in vocab
        # Sort special tokens by length (descending) to handle overlapping tokens correctly
        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.byte_to_id:
                new_id = max(self.vocab.keys()) + 1
                self.vocab[new_id] = token_bytes
                self.byte_to_id[token_bytes] = new_id
        
        # Build merge lookup table for faster encoding
        self.merge_ranks = {merge: i for i, merge in enumerate(merges)}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "BPETokenizer":
        """Create tokenizer from vocabulary and merges files."""
        # Load vocabulary
        with open(vocab_filepath) as f:
            vocab_data = json.load(f)
            vocab = {int(k): bytes(v) for k, v in vocab_data.items()}
        
        # Load merges
        merges = []
        with open(merges_filepath) as f:
            for line in f:
                first, second = line.strip().split()
                merges.append((first.encode("utf-8"), second.encode("utf-8")))
        
        return cls(vocab, merges, special_tokens)


# ============================================================================
# CORE BPE PROCESSING METHODS
# ============================================================================

    def _apply_merges_to_token(self, token: bytes) -> List[bytes]:
        """Apply BPE merges to a single token using learned merge rules.
        
        This function takes a raw token (sequence of bytes) and applies the BPE 
        merge rules learned during training to create subword units.
        
        Algorithm:
        1. Start with individual bytes as the initial segmentation
        2. Repeatedly find the highest-priority merge (lowest rank number)
        3. Apply the merge to combine adjacent segments
        4. Continue until no more merges are possible
        
        Args:
            token: Raw token as bytes (e.g., b'hello')
            
        Returns:
            List of merged byte sequences (subwords)
            
        Example:
            token = b'hello'
            merge_ranks = {(b'h', b'e'): 0, (b'l', b'l'): 1, (b'he', b'llo'): 2}
            
            Step 1: parts = [b'h', b'e', b'l', b'l', b'o']
            Step 2: Find best merge - (b'h', b'e') has rank 0 (highest priority)
            Step 3: parts = [b'he', b'l', b'l', b'o']
            Step 4: Find best merge - (b'l', b'l') has rank 1
            Step 5: parts = [b'he', b'll', b'o']  
            Step 6: Find best merge - (b'he', b'll') not in ranks, no merge for (b'll', b'o')
            Result: [b'he', b'll', b'o']
        """
        # Step 1: Convert token into list of individual bytes
        # Example: b'hello' → [b'h', b'e', b'l', b'l', b'o']
        parts = [bytes([b]) for b in token]
        
        # Step 2: Apply merges iteratively until no more merges possible
        while len(parts) > 1:
            # Find the highest-priority merge (lowest rank number)
            best_rank = float('inf')  # Start with worst possible rank
            best_idx = None          # Index where best merge should be applied
            
            # Check all adjacent pairs for possible merges
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                
                # Check if this pair has a learned merge rule
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    
                    # Lower rank = higher priority (learned earlier in training)
                    if rank < best_rank:
                        best_rank = rank
                        best_idx = i
            
            # If no merge found, we're done
            if best_idx is None:
                break
                
            # Step 3: Apply the best merge
            # Combine parts[best_idx] and parts[best_idx + 1]
            merged = parts[best_idx] + parts[best_idx + 1]
            
            # Reconstruct parts list with the merge applied
            # Before: [..., part1, part2, part3, ...]
            # After:  [..., merged_part, part3, ...]
            parts = parts[:best_idx] + [merged] + parts[best_idx + 2:]
        
        return parts


# ============================================================================
# AUTOMATIC ENCODING INTERFACE
# ============================================================================

    def encode_auto(self, text: str, strategy: str = "auto", verbose: bool = False) -> List[int]:
        """Automatically choose the best encoding strategy based on text size.
        
        Args:
            text: Input text to encode
            strategy: Encoding strategy:
                - "auto": Automatically choose based on text length (recommended)
                - "simple": Always use simple encode() method  
                - "iterable": Use streaming encode_iterable() method
                - "large_memory": Use memory-efficient approach for very large texts
            verbose: Whether to print strategy selection details
            
        Returns:
            List of token IDs
        """
        text_length = len(text)
        
        # Determine strategy
        if strategy == "auto":
            strategy = TokenizerConfig.get_text_strategy(text_length)
        
        if verbose:
            print(f"Encoding {TokenizerConfig.format_size(text_length, 'chars')} using '{strategy}' strategy")
        
        if strategy == "simple":
            return self._encode_simple(text)
        elif strategy == "iterable":
            # Split text into chunks and use iterable approach
            chunk_size = TokenizerConfig.BUFFER_SIZE_CHARS
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            return list(self.encode_iterable(chunks))
        elif strategy == "large_memory":
            # For very large texts, write to temp file and use file-based approach
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
                f.write(text)
                temp_path = f.name
            
            try:
                return list(self.encode_large_file(temp_path))
            finally:
                os.unlink(temp_path)  # Clean up temp file
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def encode_file_auto(self, file_path: str | os.PathLike, strategy: str = "auto", verbose: bool = False) -> Iterator[int]:
        """Automatically choose the best file encoding strategy based on file size.
        
        Args:
            file_path: Path to file to encode
            strategy: Encoding strategy:
                - "auto": Automatically choose based on file size (recommended)
                - "simple_file": Read entire file and use simple encode()
                - "large_file": Use parallel processing for large files
            verbose: Whether to print strategy selection details
            
        Yields:
            Token IDs one at a time
        """
        # Check file size
        file_size = os.path.getsize(file_path)
        
        # Determine strategy
        if strategy == "auto":
            strategy = TokenizerConfig.get_file_strategy(file_size)
        
        if verbose:
            print(f"Encoding file {TokenizerConfig.format_size(file_size, 'bytes')} using '{strategy}' strategy")
        
        if strategy == "simple_file":
            # Read entire file and use simple encoding
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            yield from self._encode_simple(text)
        elif strategy == "large_file":
            # Use parallel processing for large files
            yield from self._encode_large_file(file_path)
        else:
            raise ValueError(f"Unknown file strategy: {strategy}")


# ============================================================================
# CORE ENCODING IMPLEMENTATIONS
# ============================================================================

    def encode(self, text: str) -> List[int]:
        """Convenience method that automatically selects encoding strategy.
        
        This is the main public interface. For explicit strategy control,
        use encode_auto() or encode_file_auto().
        """
        return self.encode_auto(text, strategy="auto")

    def _encode_simple(self, text: str) -> List[int]:
        """Encode text into a sequence of token IDs using the simple encode() method.
        
        This method is optimized for small to medium-sized texts.
        It processes the entire text at once with full BPE encoding.
        
        The encoding process has two phases:
        1. Handle special tokens first (they have priority)
        2. Process remaining text with BPE tokenization
        
        Args:
            text: Input text to encode
            
        Returns:
            List of token IDs
            
        Example:
            text = "Hello<|endoftext|>World"
            special_tokens = ["<|endoftext|>"]
            
            Process:
            1. "Hello" - no special token at start → process as regular text
            2. "<|endoftext|>" - found special token → encode as single token
            3. "World" - no special token at start → process as regular text
        """
        result = []
        current_text = text  # Working copy of remaining text to process
        
        # MAIN LOOP: Process text from left to right
        while current_text:
            found_special = False
            
            # PHASE 1: Check if current position starts with a special token
            # We check special tokens first because they have higher priority than BPE
            for special in self.special_tokens:
                if current_text.startswith(special):
                    # Found a special token at the current position!
                    # Example: current_text = "<|endoftext|>Hello"
                    #          special = "<|endoftext|>"
                    #          startswith() returns True
                    
                    # Encode the special token as a single unit
                    token_bytes = special.encode("utf-8")
                    result.append(self.byte_to_id[token_bytes])
                    
                    # Remove the special token from the text
                    # Example: current_text becomes "Hello"
                    current_text = current_text[len(special):]
                    
                    found_special = True
                    break  # Stop checking other special tokens
            
            # PHASE 2: If no special token found, process regular text
            if not found_special:
                # Find the position of the NEXT special token (if any)
                # We need to know how much regular text to process before
                # hitting another special token
                next_special_pos = len(current_text)  # Default: process all remaining text
                
                for special in self.special_tokens:
                    pos = current_text.find(special)  # Find special token anywhere in text
                    if pos != -1:  # Found a special token later in the text
                        # Use the earliest special token position
                        # Example: current_text = "Hello<|endoftext|>World<|pad|>"
                        #          pos for "<|endoftext|>" = 5
                        #          pos for "<|pad|>" = 18
                        #          next_special_pos = min(18, 5) = 5
                        next_special_pos = min(next_special_pos, pos)
                
                # Extract the chunk of regular text to process
                # Example: current_text = "Hello<|endoftext|>World"
                #          next_special_pos = 5
                #          chunk = "Hello"
                chunk = current_text[:next_special_pos]
                
                if chunk:  # Only process if there's actual text
                    # Use BPE tokenization for regular text
                    # Pass empty list [] for special_tokens because we handle them separately
                    tokens = process_chunk_basic(chunk, [])
                    
                    # Apply BPE merges to each token
                    for token in tokens:
                        # Apply learned merges (e.g., "th" + "e" → "the")
                        merged_parts = self._apply_merges_to_token(token)
                        
                        # Convert merged parts to token IDs
                        for part in merged_parts:
                            if part in self.byte_to_id:
                                result.append(self.byte_to_id[part])
                            else:
                                # Fallback: split unknown tokens into individual bytes
                                for b in part:
                                    result.append(self.byte_to_id[bytes([b])])
                
                # Move past the processed text
                # Example: current_text = "Hello<|endoftext|>World"
                #          next_special_pos = 5
                #          current_text becomes "<|endoftext|>World"
                current_text = current_text[next_special_pos:]
        
        return result

    def _encode_large_file(self, file_path: str | os.PathLike, chunk_size_mb: int = None) -> Iterator[int]:
        """Internal method to encode large files using parallel processing."""
        if chunk_size_mb is None:
            chunk_size_mb = TokenizerConfig.CHUNK_SIZE_MB
        return self.encode_large_file(file_path, chunk_size_mb)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an iterable of strings into token IDs using streaming approach.
        
        This method is designed for processing large files or streams where you 
        can't load all text into memory at once. It maintains a buffer and only
        processes complete tokens to avoid splitting tokens across chunk boundaries.
        
        Key Benefits:
        - Memory efficient: processes chunks incrementally
        - Handles large files: no need to load entire file into memory  
        - Preserves token boundaries: never splits tokens across chunks
        - Streaming output: yields results as they're computed
        
        Algorithm:
        1. Accumulate text chunks in a buffer
        2. Find complete token boundaries using GPT-2 regex
        3. Process only complete tokens, keep incomplete ones for next iteration
        4. Yield token IDs one by one
        
        Args:
            iterable: Iterator yielding strings to encode (e.g., file chunks)
            
        Yields:
            Token IDs one at a time
            
        Example:
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
        """
        buffer = ""  # Accumulate text chunks
        
        for chunk in iterable:
            # Add new chunk to buffer
            buffer += chunk
            
            # Find the last complete token boundary using GPT-2 regex
            # We need to avoid splitting tokens across chunk boundaries
            last_complete = 0
            
            # Use PAT (GPT-2 regex) to find all token boundaries
            for match in PAT.finditer(buffer):
                # Only consider tokens that end before the buffer end
                # This ensures we don't split incomplete tokens
                if match.end() < len(buffer):
                    last_complete = match.end()
                # If match.end() == len(buffer), the token might be incomplete
                # so we don't include it in this iteration
            
            # Process complete tokens if any found
            if last_complete > 0:
                # Extract text with complete tokens only
                complete_text = buffer[:last_complete]
                
                # Encode complete tokens and yield results
                for token_id in self.encode(complete_text):
                    yield token_id
                    
                # Keep remainder (incomplete tokens) for next iteration    
                buffer = buffer[last_complete:]
        
        # Process any remaining text in buffer (end of stream)
        if buffer:
            for token_id in self.encode(buffer):
                yield token_id

    def encode_large_file(self, file_path: str | os.PathLike, chunk_size_mb: int = 100) -> Iterator[int]:
        """Encode very large files efficiently using parallel processing and chunking.
        
        This method is optimized for preprocessing large corpora during model training.
        It uses the same parallel processing techniques as BPE training to handle
        files that are too large for regular encode() method.
        
        Key Features:
        - Parallel processing: uses multiple CPU cores
        - Memory efficient: processes file in chunks
        - Special token aware: respects special token boundaries
        - Streaming output: yields token IDs as computed
        - Automatic strategy: chooses best approach based on file size
        
        When to use:
        - Preprocessing large datasets for training (> 100MB files)
        - Batch tokenization of multiple large documents
        - Memory-constrained environments with large texts
        
        Args:
            file_path: Path to the file to encode
            chunk_size_mb: Size of each chunk in MB (default: 100MB)
            
        Yields:
            Token IDs one at a time
            
        Example:
            # Tokenize a 5GB dataset efficiently
            for token_id in tokenizer.encode_large_file("huge_dataset.txt"):
                process(token_id)  # Process tokens as they're generated
        """
        # Import parallel processing functions from train_bpe
        from .train_bpe import auto_tokenize, BPEConfig
        
        # Check file size to determine strategy
        with open(file_path, 'rb') as f:
            file_size = f.seek(0, 2)
        
        print(f"Encoding large file: {BPEConfig.format_size(file_size)}")
        
        # For very large files, use parallel processing from train_bpe
        if file_size > 100 * 1024 * 1024:  # > 100MB
            # Use the advanced parallel tokenization from train_bpe
            pre_tokens = auto_tokenize(file_path, self.special_tokens, strategy="auto")
            
            # Convert pre-tokens to final token IDs using BPE merges
            for token in pre_tokens:
                # Apply BPE merges to each pre-token
                merged_parts = self._apply_merges_to_token(token)
                
                # Convert merged parts to token IDs
                for part in merged_parts:
                    if part in self.byte_to_id:
                        yield self.byte_to_id[part]
                    else:
                        # Fallback: split unknown tokens into individual bytes
                        for b in part:
                            yield self.byte_to_id[bytes([b])]
        else:
            # For smaller files, use chunked reading with regular encode
            chunk_size_bytes = chunk_size_mb * 1024 * 1024
            
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                buffer = ""
                
                while True:
                    chunk = f.read(chunk_size_bytes)
                    if not chunk:
                        break
                    
                    buffer += chunk
                    
                    # Find last complete token boundary
                    last_complete = 0
                    for match in PAT.finditer(buffer):
                        if match.end() < len(buffer):
                            last_complete = match.end()
                    
                    if last_complete > 0:
                        # Process complete portion
                        for token_id in self.encode(buffer[:last_complete]):
                            yield token_id
                        buffer = buffer[last_complete:]
                
                # Process remaining buffer
                if buffer:
                    for token_id in self.encode(buffer):
                        yield token_id

 

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back into text.
        
        Args:
            ids: List of token IDs to decode
            
        Returns:
            Decoded text string
        """
        # Convert IDs to bytes and concatenate
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
            else:
                # Handle unknown token IDs with replacement character
                byte_parts.append("".encode("utf-8"))
        
        # Decode bytes to string, replacing invalid sequences with replacement character
        return b"".join(byte_parts).decode("utf-8", errors="replace")


# ============================================================================
# USAGE EXAMPLES AND DEMONSTRATIONS
# ============================================================================

if __name__ == "__main__":
    """
    Comprehensive demonstration of the reorganized tokenizer with automatic strategy selection.
    """
    
    print("=== BPE Tokenizer with Automatic Strategy Selection ===\n")
    
    # Example tokenizer initialization (would normally load from files)
    vocab = {i: bytes([i]) for i in range(256)}  # Basic byte vocabulary
    merges = [(b'h', b'e'), (b'l', b'l'), (b't', b'h')]  # Example merges
    special_tokens = ["<|endoftext|>", "<|pad|>"]
    
    tokenizer = BPETokenizer(vocab, merges, special_tokens)
    
    # =========================================================================
    # TEXT ENCODING - AUTOMATIC STRATEGY SELECTION
    # =========================================================================
    
    print("1. AUTOMATIC TEXT ENCODING")
    print("-" * 40)
    
    # Small text - uses simple strategy
    small_text = "Hello, world!"
    tokens_small = tokenizer.encode_auto(small_text, verbose=True)
    print(f"Input: '{small_text}'")
    print(f"Tokens: {tokens_small[:10]}...\n")
    
    # Medium text - uses iterable strategy  
    medium_text = "Hello, world! " * 1000  # ~14K characters
    tokens_medium = tokenizer.encode_auto(medium_text, verbose=True)
    print(f"Input: {len(medium_text)} characters")
    print(f"First 10 tokens: {tokens_medium[:10]}\n")
    
    # Large text - uses memory-efficient strategy
    large_text = "Hello, world! " * 100_000  # ~1.4M characters
    tokens_large = tokenizer.encode_auto(large_text, verbose=True)
    print(f"Input: {len(large_text)} characters")
    print(f"First 10 tokens: {tokens_large[:10]}\n")
    
    # =========================================================================
    # FILE ENCODING - AUTOMATIC STRATEGY SELECTION  
    # =========================================================================
    
    print("2. AUTOMATIC FILE ENCODING")
    print("-" * 40)
    
    # Create sample files of different sizes
    import tempfile
    import os
    
    # Small file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Hello, world! " * 100)  # ~1.4KB
        small_file = f.name
    
    # Large file 
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write("Hello, world! " * 1_000_000)  # ~14MB
        large_file = f.name
    
    try:
        # Small file encoding
        print("Small file:")
        tokens_iter = tokenizer.encode_file_auto(small_file, verbose=True)
        small_tokens = list(tokens_iter)
        print(f"Encoded {len(small_tokens)} tokens\n")
        
        # Large file encoding
        print("Large file:")
        tokens_iter = tokenizer.encode_file_auto(large_file, verbose=True)
        large_tokens = []
        for i, token_id in enumerate(tokens_iter):
            large_tokens.append(token_id)
            if i >= 10:  # Just show first 10 for demo
                break
        print(f"First 10 tokens: {large_tokens}\n")
        
    finally:
        # Clean up temp files
        os.unlink(small_file)
        os.unlink(large_file)
    
    # =========================================================================
    # MANUAL STRATEGY SELECTION
    # =========================================================================
    
    print("3. MANUAL STRATEGY SELECTION")
    print("-" * 40)
    
    test_text = "Hello, world! " * 1000
    
    strategies = ["simple", "iterable", "large_memory"]
    for strategy in strategies:
        tokens = tokenizer.encode_auto(test_text, strategy=strategy, verbose=True)
        print(f"Strategy '{strategy}': {len(tokens)} tokens\n")
    
    # =========================================================================
    # CONFIGURATION OVERVIEW
    # =========================================================================
    
    print("4. CONFIGURATION THRESHOLDS")
    print("-" * 40)
    
    print("Text Size Thresholds:")
    print(f"  Small text:   < {TokenizerConfig.format_size(TokenizerConfig.SMALL_TEXT_THRESHOLD, 'chars')}")
    print(f"  Medium text:  < {TokenizerConfig.format_size(TokenizerConfig.MEDIUM_TEXT_THRESHOLD, 'chars')}")
    print(f"  Large text:   ≥ {TokenizerConfig.format_size(TokenizerConfig.LARGE_TEXT_THRESHOLD, 'chars')}")
    print()
    
    print("File Size Thresholds:")
    print(f"  Small file:   < {TokenizerConfig.format_size(TokenizerConfig.SMALL_FILE_THRESHOLD, 'bytes')}")
    print(f"  Large file:   ≥ {TokenizerConfig.format_size(TokenizerConfig.LARGE_FILE_THRESHOLD, 'bytes')}")
    print()
    
    print("Performance Settings:")
    print(f"  Chunk size:   {TokenizerConfig.CHUNK_SIZE_MB}MB")
    print(f"  Buffer size:  {TokenizerConfig.format_size(TokenizerConfig.BUFFER_SIZE_CHARS, 'chars')}")
    
    print("\n=== Demo Complete ===")


# ============================================================================
# PERFORMANCE COMPARISON GUIDE
# ============================================================================

"""
PERFORMANCE COMPARISON GUIDE
============================

Text Size | Strategy      | Memory Usage | Speed    | Best For
----------|---------------|--------------|----------|------------------
< 10K     | simple        | Low          | Fast     | Real-time inference
10K-1M    | iterable      | Medium       | Medium   | Document processing  
> 1M      | large_memory  | Low          | Fast     | Batch preprocessing

File Size | Strategy      | Cores | Memory | Speed    | Best For
----------|---------------|-------|--------|----------|------------------
< 1MB     | simple_file   | 1     | Medium | Fast     | Small documents
> 100MB   | large_file    | 8     | Low    | V.Fast   | Training corpus

USAGE RECOMMENDATIONS
====================

1. **Real-time Inference**:
   ```python
   # Use default encode() - automatically selects best strategy
   tokens = tokenizer.encode(user_input)
   ```

2. **Document Processing**:
   ```python
   # Process multiple documents efficiently
   for doc in documents:
       tokens = tokenizer.encode_auto(doc, strategy="auto", verbose=True)
   ```

3. **Large Dataset Preprocessing**:
   ```python
   # Process huge training files
   for token_id in tokenizer.encode_file_auto("huge_corpus.txt"):
       training_batch.append(token_id)
   ```

4. **Memory-Constrained Environments**:
   ```python
   # Force memory-efficient strategies
   tokens = tokenizer.encode_auto(text, strategy="iterable") 
   ```

5. **Custom Thresholds**:
   ```python
   # Adjust thresholds for your use case
   TokenizerConfig.SMALL_TEXT_THRESHOLD = 5_000  # More aggressive
   TokenizerConfig.LARGE_FILE_THRESHOLD = 50 * 1024 * 1024  # 50MB
   ```
"""
