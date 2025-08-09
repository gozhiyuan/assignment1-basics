import os
import regex as re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Any, BinaryIO, Iterator, Callable
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import cProfile
import pstats
import time
from functools import wraps


# ============================================================================
# PROFILING UTILITIES
# ============================================================================

class ProfilingContext:
    """Context manager for profiling and timing operations."""
    
    def __init__(self):
        self.timings = {}
        self.step_count = 0
    
    def time_operation(self, name: str):
        """Decorator to time operations."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                duration = end_time - start_time
                self.timings[name] = duration
                print(f"⏱️  {name}: {duration:.2f}s")
                return result
            return wrapper
        return decorator
    
    def print_summary(self):
        """Print timing summary."""
        print("\n" + "="*50)
        print("TIMING SUMMARY")
        print("="*50)
        total_time = sum(self.timings.values())
        for name, duration in sorted(self.timings.items(), key=lambda x: x[1], reverse=True):
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            print(f"{name:<30}: {duration:>8.2f}s ({percentage:>5.1f}%)")
        print(f"{'Total':<30}: {total_time:>8.2f}s (100.0%)")
        print("="*50)

# Global profiling context
profiling_ctx = ProfilingContext()


# ============================================================================
# CONFIGURATION SETTINGS
# ============================================================================

class BPEConfig:
    """Global configuration for BPE training with size thresholds and processing settings."""
    
    # File size thresholds (in bytes)
    SMALL_FILE_THRESHOLD = 1024 * 1024          # 1MB - use sequential processing
    MEDIUM_FILE_THRESHOLD = 100 * 1024 * 1024   # 100MB - use parallel list processing
    LARGE_FILE_THRESHOLD = 1024 * 1024 * 1024   # 1GB - use parallel iterator
    VERY_LARGE_FILE_THRESHOLD = 5 * 1024 * 1024 * 1024   # 5GB - use parallel streaming
    
    # Memory-efficient trainer threshold (for files that cause memory issues)
    MEMORY_EFFICIENT_THRESHOLD = 4 * 1024 * 1024 * 1024  # 2GB - use streaming trainer
    
    # Chunk processing thresholds
    LARGE_CHUNK_THRESHOLD = 100 * 1024 * 1024   # 100MB - use memory-efficient chunk processing
    LARGE_TEXT_THRESHOLD = 1024 * 1024          # 1MB - use finditer instead of findall
    
    # Processing settings
    MINI_CHUNK_SIZE = 4096                      # 4KB - for boundary finding
    MULTIPROCESSING_CHUNKSIZE = 1               # Chunk size for pool.imap
    
    # Memory management settings (tuned for ~16-32GB systems)
    MAX_WORKERS_FOR_LARGE_FILES = 6             # Limit workers for large files to avoid OOM. Will be reduced for very large files.
    MAX_CHUNK_SIZE_PER_WORKER = 200 * 1024 * 1024  # 200MB max per worker (reduced from 400MB)
    ULTRA_LARGE_FILE_THRESHOLD = 8 * 1024 * 1024 * 1024  # 8GB - force single worker for ultra large files
    
    @classmethod
    def get_strategy_for_file_size(cls, file_size: int) -> str:
        """Determine the optimal tokenization strategy based on file size."""
        if file_size < cls.SMALL_FILE_THRESHOLD:
            return "sequential"
        elif file_size < cls.MEDIUM_FILE_THRESHOLD:
            return "parallel_list"
        elif file_size < cls.LARGE_FILE_THRESHOLD:
            return "parallel_iter"
        elif file_size < cls.VERY_LARGE_FILE_THRESHOLD:
            return "parallel_streaming"
        else:
            return "parallel_streaming"  # Use streaming for very large files too
    
    @classmethod
    def should_use_memory_efficient_trainer(cls, file_size: int) -> bool:
        """Determine if memory-efficient trainer should be used."""
        return file_size > cls.MEMORY_EFFICIENT_THRESHOLD
    
    @classmethod
    def get_optimal_workers(cls, file_size: int) -> int:
        """Calculate optimal number of workers based on file size to avoid OOM."""
        # Allow manual override via environment variable
        manual_workers = os.environ.get('BPE_MAX_WORKERS')
        if manual_workers:
            try:
                return max(1, int(manual_workers))
            except ValueError:
                pass
        
        # For ultra large files, force single worker to avoid OOM
        if file_size > cls.ULTRA_LARGE_FILE_THRESHOLD:
            return 1
        
        # Start with available cores minus 1, but cap at reasonable number
        max_workers = max(1, min(mp.cpu_count() - 1, 8))  # Never use more than 8 workers
        
        # For large files (>= 1GB), apply strict limits to avoid OOM
        if file_size >= cls.LARGE_FILE_THRESHOLD:
            # Calculate workers based on memory per worker to avoid huge chunks
            estimated_chunks = max(1, file_size // cls.MAX_CHUNK_SIZE_PER_WORKER)
            
            # Apply all limits: CPU count, memory-based calculation, and our hard limit
            max_workers = min(
                max_workers,
                cls.MAX_WORKERS_FOR_LARGE_FILES,
                estimated_chunks,
                4  # Additional conservative limit for files >= 1GB
            )
        
        # For very large files (>= 2GB), be even more conservative
        if file_size >= cls.MEMORY_EFFICIENT_THRESHOLD:
            max_workers = min(max_workers, 2)  # Max 2 workers for 2GB+ files
        
        return max(1, max_workers)
    
    @classmethod
    def format_size(cls, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes < 1024 * 1024:
            return f"{size_bytes // 1024}KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes // (1024 * 1024)}MB"
        else:
            return f"{size_bytes // (1024 * 1024 * 1024)}GB"


# ============================================================================
# CONSTANTS AND PATTERNS
# ============================================================================

# Pre-tokenization regex pattern from GPT-2
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


# ============================================================================
# CORE UTILITY FUNCTIONS
# ============================================================================

def get_initial_vocab() -> Dict[int, bytes]:
    """Initialize vocabulary with all possible bytes (0-255)."""
    return {i: bytes([i]) for i in range(256)}


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_tokens: List[bytes]
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    
    Args:
        file: Binary file to chunk
        desired_num_chunks: Target number of chunks
        split_special_tokens: List of special tokens (as bytes) to avoid splitting
    """
    assert isinstance(split_special_tokens, list), (
        "split_special_tokens must be a list of bytes"
    )
    assert all(isinstance(token, bytes) for token in split_special_tokens), (
        "All special tokens must be represented as bytestrings"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        
        while True:
            mini_chunk = file.read(BPEConfig.MINI_CHUNK_SIZE)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find any of the special tokens in the mini chunk
            found_positions = []
            for special_token in split_special_tokens:
                found_at = mini_chunk.find(special_token)
                if found_at != -1:
                    found_positions.append(found_at)
            
            # If we found any special token, use the earliest one
            if found_positions:
                chunk_boundaries[bi] = initial_position + min(found_positions)
                break
                
            initial_position += BPEConfig.MINI_CHUNK_SIZE

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


# ============================================================================
# TOKENIZATION FUNCTIONS
# ============================================================================

class TokenizationStrategy:
    """Enumeration of available tokenization strategies."""
    SEQUENTIAL = "sequential"      # Single-threaded, lowest memory
    PARALLEL_LIST = "parallel_list"    # Multi-threaded with lists, fastest but high memory
    PARALLEL_ITER = "parallel_iter"    # Multi-threaded with iterators, balanced
    PARALLEL_STREAMING = "parallel_streaming"  # Multi-threaded streaming, lowest memory


def process_chunk_basic(chunk_text: str, special_tokens: List[str]) -> List[bytes]:
    """Basic chunk processing function - returns list of tokens.
    
    Used by: PARALLEL_LIST strategy
    """
    # Fast path: if no special tokens, use GPT-2 tokenization pattern
    if not special_tokens:
        # For large chunks, use finditer for memory efficiency
        if len(chunk_text) > BPEConfig.LARGE_TEXT_THRESHOLD:  # > 1MB chunk
            return [match.group().encode("utf-8") for match in PAT.finditer(chunk_text)]
        else:
            return [token.encode("utf-8") for token in PAT.findall(chunk_text)]

    # Handle special tokens
    special_pattern = f"({'|'.join(map(re.escape, special_tokens))})"
    special_chunks = re.split(special_pattern, chunk_text)

    pre_tokens = []
    for chunk in special_chunks:
        if not chunk:
            continue
            
        if chunk in special_tokens:
            pre_tokens.append(chunk.encode("utf-8"))
        else:
            # For memory efficiency with large chunks
            if len(chunk) > BPEConfig.LARGE_TEXT_THRESHOLD:  # > 1MB sub-chunk
                regular_tokens = [match.group().encode("utf-8") for match in PAT.finditer(chunk)]
                pre_tokens.extend(regular_tokens)
            else:
                regular_tokens = PAT.findall(chunk)
                pre_tokens.extend(token.encode("utf-8") for token in regular_tokens)

    return pre_tokens


def process_chunk_iter(chunk_text: str, special_tokens: List[str]) -> Iterator[bytes]:
    """Iterator chunk processing function - yields tokens one by one.
    
    Used by: PARALLEL_ITER and PARALLEL_STREAMING strategies
    """
    # Fast path: if no special tokens, use GPT-2 tokenization pattern
    if not special_tokens:
        for match in PAT.finditer(chunk_text):
            yield match.group().encode("utf-8")
        return

    # Handle special tokens
    special_pattern = f"({'|'.join(map(re.escape, special_tokens))})"
    special_chunks = re.split(special_pattern, chunk_text)

    for chunk in special_chunks:
        if not chunk:
            continue
            
        if chunk in special_tokens:
            yield chunk.encode("utf-8")
        else:
            for match in PAT.finditer(chunk):
                yield match.group().encode("utf-8")


def process_chunk_bytes(chunk: bytes, special_tokens: List[str]) -> List[bytes]:
    """Process a chunk of bytes into tokens."""
    text = chunk.decode('utf-8', errors='replace')
    return process_chunk_basic(text, special_tokens)


# ============================================================================
# PARALLEL PROCESSING IMPLEMENTATIONS
# ============================================================================

def tokenize_sequential(input_path: str | os.PathLike, special_tokens: List[str]) -> Iterator[bytes]:
    """Sequential processing - single threaded, minimal memory usage."""
    with open(input_path, 'rb') as f:
        chunk = f.read()
        text = chunk.decode('utf-8', errors='replace')
        yield from process_chunk_iter(text, special_tokens)


def tokenize_parallel_list(input_path: str | os.PathLike, special_tokens: List[str]) -> List[bytes]:
    """Parallel processing with lists - fastest but highest memory usage."""
    # Get file size for optimal worker calculation
    with open(input_path, 'rb') as f:
        file_size = f.seek(0, 2)
    n_cores = BPEConfig.get_optimal_workers(file_size)
    print(f"Using {n_cores} workers for parallel list processing")
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, desired_num_chunks=n_cores, split_special_tokens=[s.encode("utf-8") for s in special_tokens]
        )
    
    chunks = []
    chunk_sizes = []
    with open(input_path, 'rb') as f:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_data = f.read(end - start)
            chunks.append(chunk_data)
            chunk_sizes.append(len(chunk_data))
    
    max_chunk_size = max(chunk_sizes) if chunk_sizes else 0
    
    # Convert chunks to text for processing
    text_chunks = [(chunk.decode('utf-8', errors='replace'), special_tokens) for chunk in chunks]
    
    # Use memory-efficient processing for very large chunks
    if max_chunk_size > BPEConfig.LARGE_CHUNK_THRESHOLD:
        print(f"Using memory-efficient processing for large chunks (max: {BPEConfig.format_size(max_chunk_size)})")
        # For very large chunks, use the iterator approach and convert to list
        with mp.Pool(n_cores) as pool:
            results = pool.starmap(_process_chunk_iter_wrapper, text_chunks)
    else:
        # For normal chunks, use the basic approach
        with mp.Pool(n_cores) as pool:
            results = pool.starmap(process_chunk_basic, text_chunks)
    
    return [token for chunk_tokens in results for token in chunk_tokens]


def _process_chunk_iter_wrapper(text, special_tokens):
    """Wrapper function for process_chunk_iter that can be pickled for multiprocessing."""
    return list(process_chunk_iter(text, special_tokens))


def _process_chunk_range_iter(args):
    """Process a chunk by reading it from file on demand."""
    input_path, start, end, special_tokens = args
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk_data = f.read(end - start)
        text = chunk_data.decode('utf-8', errors='replace')
        return list(process_chunk_iter(text, special_tokens))


def _process_chunk_range_streaming(args):
    """Process a chunk for streaming tokenization."""
    input_path, start, end, special_tokens = args
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk_data = f.read(end - start)
        text = chunk_data.decode('utf-8', errors='replace')
        return list(process_chunk_iter(text, special_tokens))


def tokenize_parallel_iter(input_path: str | os.PathLike, special_tokens: List[str]) -> Iterator[bytes]:
    """Parallel processing with iterators - balanced speed and memory."""
    # Get file size for optimal worker calculation
    with open(input_path, 'rb') as f:
        file_size = f.seek(0, 2)
    n_cores = BPEConfig.get_optimal_workers(file_size)
    print(f"Using {n_cores} workers for parallel iter processing")
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, desired_num_chunks=n_cores, split_special_tokens=[s.encode("utf-8") for s in special_tokens]
        )
    
    # Create chunk arguments without loading data into memory
    chunk_args = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    
    print(f"Processing {len(chunk_args)} chunks in parallel...")
    
    with mp.Pool(n_cores) as pool:
        for chunk_results in pool.imap(
            _process_chunk_range_iter, 
            chunk_args, 
            chunksize=BPEConfig.MULTIPROCESSING_CHUNKSIZE
        ):
            yield from chunk_results


def tokenize_parallel_streaming(input_path: str | os.PathLike, special_tokens: List[str]) -> Iterator[bytes]:
    """Ultra memory-efficient parallel processing with streaming."""
    # Get file size for optimal worker calculation
    with open(input_path, 'rb') as f:
        file_size = f.seek(0, 2)
    n_cores = BPEConfig.get_optimal_workers(file_size)
    print(f"Using {n_cores} workers for parallel streaming (file size: {BPEConfig.format_size(file_size)})")
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, desired_num_chunks=n_cores, split_special_tokens=[s.encode("utf-8") for s in special_tokens]
        )
    
    chunk_args = [
        (input_path, start, end, special_tokens) 
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    
    print(f"Processing {len(chunk_args)} chunks in parallel (streaming)...")
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        future_to_chunk = {
            executor.submit(_process_chunk_range_streaming, args): i 
            for i, args in enumerate(chunk_args)
        }
        
        for future in as_completed(future_to_chunk):
            chunk_results = future.result()
            yield from chunk_results


# ============================================================================
# MAIN TOKENIZATION INTERFACE
# ============================================================================

def auto_tokenize(
    input_path: str | os.PathLike, 
    special_tokens: List[str], 
    strategy: str = "auto"
):
    """Main interface for tokenization with automatic strategy selection.
    
    Args:
        input_path: Path to input corpus
        special_tokens: List of special tokens to preserve
        strategy: Tokenization strategy:
            - "auto": Automatically choose based on file size
            - "sequential": Single-threaded, minimal memory
            - "parallel_list": Multi-threaded with lists, fastest
            - "parallel_iter": Multi-threaded with iterators, balanced
            - "parallel_streaming": Multi-threaded streaming, lowest memory
    
    Returns:
        Iterator or List of pre-tokenized byte sequences
    """
    with open(input_path, 'rb') as f:
        file_size = f.seek(0, 2)
    
    size_mb = file_size // (1024 * 1024)
    
    if strategy == "auto":
        if file_size < BPEConfig.SMALL_FILE_THRESHOLD:  # < 1MB
            strategy = TokenizationStrategy.SEQUENTIAL
        elif file_size < BPEConfig.MEDIUM_FILE_THRESHOLD:  # < 100MB
            strategy = TokenizationStrategy.PARALLEL_LIST
        elif file_size < BPEConfig.LARGE_FILE_THRESHOLD:  # < 1GB
            strategy = TokenizationStrategy.PARALLEL_ITER
        else:  # >= 1GB
            strategy = TokenizationStrategy.PARALLEL_STREAMING
    
    print(f"Using {strategy} strategy for {BPEConfig.format_size(file_size)} corpus")
    
    if strategy == TokenizationStrategy.SEQUENTIAL:
        return tokenize_sequential(input_path, special_tokens)
    elif strategy == TokenizationStrategy.PARALLEL_LIST:
        return tokenize_parallel_list(input_path, special_tokens)
    elif strategy == TokenizationStrategy.PARALLEL_ITER:
        return tokenize_parallel_iter(input_path, special_tokens)
    elif strategy == TokenizationStrategy.PARALLEL_STREAMING:
        return tokenize_parallel_streaming(input_path, special_tokens)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================================
# BPE TRAINER CLASSES
# ============================================================================

class BPETrainerMemoryEfficient:
    """Memory-efficient BPE trainer that works with word iterators for very large corpora.
    
    Instead of storing all words in memory, this trainer:
    1. Builds initial pair counts by streaming through words once
    2. Re-processes the corpus for each merge step
    3. Uses significantly less memory at the cost of more I/O
    """
    
    def __init__(self, vocab: Dict[int, bytes], word_source_func):
        """Initialize trainer with vocabulary and a function that yields words.
        
        Args:
            vocab: Initial vocabulary
            word_source_func: Function that when called, returns an iterator of word lists
        """
        self.vocab = vocab.copy()
        self.word_source_func = word_source_func
        self.pair_counts = Counter()
        
        # Build initial pair counts by streaming through all words once
        print("Building initial pair counts from corpus...")
        self._build_initial_pair_counts()
    
    def _build_initial_pair_counts(self):
        """Build initial pair counts by streaming through the corpus once."""
        self.pair_counts.clear()
        
        for word in self.word_source_func():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                self.pair_counts[pair] += 1
    
    def get_best_pair(self) -> Tuple[int, int]:
        """Find the most frequent pair with lexicographical tie-breaking."""
        if not self.pair_counts:
            return None
        
        # Sort pairs: primary key frequency (desc), secondary key pair bytes (desc)
        best_pair = max(
            self.pair_counts.keys(),
            key=lambda p: (
                self.pair_counts[p],
                self.vocab.get(p[0], b""),
                self.vocab.get(p[1], b""),
            ),
        )
        return best_pair
    
    def merge_pair(self, pair: Tuple[int, int]) -> None:
        """Merge a specific pair by re-processing the entire corpus.
        
        This is more expensive than the in-memory version but uses much less memory.
        """
        # Create new token
        new_token_id = len(self.vocab)
        self.vocab[new_token_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        
        # Rebuild pair counts after applying the merge
        print(f"Applying merge: {self.vocab[pair[0]]} + {self.vocab[pair[1]]} -> {self.vocab[new_token_id]}")
        self._rebuild_pair_counts_after_merge(pair, new_token_id)
    
    def _rebuild_pair_counts_after_merge(self, merged_pair: Tuple[int, int], new_token_id: int):
        """Rebuild pair counts after applying a merge by streaming through corpus."""
        self.pair_counts.clear()
        
        # Stream through corpus and apply merge, then count new pairs
        for word in self.word_source_func():
            # Apply merge to this word
            merged_word = self._apply_merge_to_word(word, merged_pair, new_token_id)
            
            # Count pairs in merged word
            for i in range(len(merged_word) - 1):
                pair = (merged_word[i], merged_word[i + 1])
                self.pair_counts[pair] += 1
    
    def _apply_merge_to_word(self, word: List[int], merged_pair: Tuple[int, int], new_token_id: int) -> List[int]:
        """Apply a merge rule to a single word."""
        if len(word) < 2:
            return word
        
        new_word = []
        i = 0
        while i < len(word):
            # Check if we can apply the merge at position i
            if (i < len(word) - 1 and 
                word[i] == merged_pair[0] and 
                word[i + 1] == merged_pair[1]):
                new_word.append(new_token_id)
                i += 2  # Skip both tokens that were merged
            else:
                new_word.append(word[i])
                i += 1
        
        return new_word


class BPETrainer:
    """Standard in-memory BPE trainer for small to medium corpora.
    
    Key Data Structures:
        - vocab: Dict[int, bytes] - Maps token IDs to byte sequences
        - words: List[List[int]] - List of words, each word is a list of token IDs
        - pair_counts: Counter - Counts frequency of each token pair
        - word_pair_indices: Dict[int, List[Tuple[int, Tuple[int, int]]]] 
          - Maps word_idx → [(position, pair), ...]
        - pair_to_words: Dict[Tuple[int, int], Set[int]]
          - Maps pair → set of word indices containing this pair
    """
    
    def __init__(self, vocab: Dict[int, bytes], words: List[List[int]]):
        self.vocab = vocab.copy()
        self.words = [word[:] for word in words]  # Deep copy
        self.pair_counts = Counter()
        self.word_pair_indices = defaultdict(list)  # word_idx -> [(pair_pos, pair), ...]
        self.pair_to_words = defaultdict(set)  # pair -> {word_indices}
        
        # Initialize pair counts and indices
        for word_idx, word in enumerate(self.words):
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                self.pair_counts[pair] += 1
                self.word_pair_indices[word_idx].append((i, pair))
                self.pair_to_words[pair].add(word_idx)
    
    def get_best_pair(self) -> Tuple[int, int]:
        """Find the most frequent pair with lexicographical tie-breaking."""
        if not self.pair_counts:
            return None
        
        best_pair = max(
            self.pair_counts.keys(),
            key=lambda p: (
                self.pair_counts[p],
                self.vocab.get(p[0], b""),
                self.vocab.get(p[1], b""),
            ),
        )
        return best_pair
    
    def merge_pair(self, pair: Tuple[int, int]) -> None:
        """Merge a specific pair and update all data structures.
        
        This is the core BPE algorithm that:
        1. Creates a new token by combining two existing tokens
        2. Updates all words that contain this pair
        3. Rebuilds pair statistics for the next iteration
        
        Args:
            pair: Tuple of (token_id1, token_id2) to merge
            
        Example:
            If pair = (108, 108) and vocab[108] = b'l', then:
            - Creates new token: vocab[256] = b'll'
            - Updates all words containing "ll" → replace with new token ID 256
            - Rebuilds pair statistics for next merge iteration
            
        Visual Example:
            Before merge (pair = (108, 108), vocab[108] = b'l'):
                word = [72, 101, 108, 108, 111]  # "Hello"
                pairs = [(72,101), (101,108), (108,108), (108,111)]
                
            After merge (new_token_id = 256, vocab[256] = b'll'):
                word = [72, 101, 256, 111]      # "He" + "ll" + "o"
                pairs = [(72,101), (101,256), (256,111)]
        """
        # Step 1: Create new token by concatenating the two tokens
        new_token_id = len(self.vocab)
        self.vocab[new_token_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        
        # Example: if pair = (108, 108) and vocab[108] = b'l'
        # Then: vocab[256] = b'l' + b'l' = b'll'
        
        # Step 2: Find all words that contain this pair
        # pair_to_words[pair] contains indices of all words with this pair
        affected_words = list(self.pair_to_words[pair])
        
        # Step 3: Process each affected word
        for word_idx in affected_words:
            old_word = self.words[word_idx]
            new_word = []
            
            # Step 3a: Remove old pair statistics for this word
            # This prevents counting the same pairs multiple times
            for pos, old_pair in self.word_pair_indices[word_idx]:
                self.pair_counts[old_pair] -= 1
                self.pair_to_words[old_pair].discard(word_idx)
                # Clean up if this pair no longer exists anywhere
                if self.pair_counts[old_pair] == 0:
                    del self.pair_counts[old_pair]
                    del self.pair_to_words[old_pair]
            
            # Step 3b: Apply the merge to create new word
            # Scan through the word and replace occurrences of the pair
            i = 0
            while i < len(old_word):
                # Check if current position has the pair we want to merge
                if (i < len(old_word) - 1 and 
                    old_word[i] == pair[0] and 
                    old_word[i + 1] == pair[1]):
                    # Replace the pair with the new token ID
                    new_word.append(new_token_id)
                    i += 2  # Skip both tokens that were merged
                else:
                    # Keep the current token as-is
                    new_word.append(old_word[i])
                    i += 1
            
            # Step 3c: Update the word and rebuild pair statistics
            self.words[word_idx] = new_word
            self.word_pair_indices[word_idx] = []
            
            # Count all new pairs in the updated word
            for i in range(len(new_word) - 1):
                new_pair = (new_word[i], new_word[i + 1])
                self.pair_counts[new_pair] += 1
                self.word_pair_indices[word_idx].append((i, new_pair))
                self.pair_to_words[new_pair].add(word_idx)


# ============================================================================
# MAIN TRAINING INTERFACE
# ============================================================================

@profiling_ctx.time_operation("Total BPE Training")
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    strategy: str = "auto",
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Main interface for BPE training with automatic optimization.

    Args:
        input_path: Path to BPE tokenizer training data
        vocab_size: Total number of items in the tokenizer's vocabulary (including special tokens)
        special_tokens: List of string special tokens to be added to the tokenizer vocabulary
        strategy: Processing strategy:
            - "auto": Automatically choose based on file size (recommended)
            - "sequential": Single-threaded, minimal memory
            - "parallel_list": Multi-threaded with lists, fastest
            - "parallel_iter": Multi-threaded with iterators, balanced
            - "parallel_streaming": Multi-threaded streaming, lowest memory

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab: The trained tokenizer vocabulary
            merges: BPE merges ordered by creation
    """
    print(f"\n🚀 Starting BPE training: {input_path}")
    print(f"Target vocab size: {vocab_size}, Special tokens: {special_tokens}")
    
    # Initialize vocabulary with bytes 0-255
    start_time = time.time()
    vocab = get_initial_vocab()
    next_token_id = len(vocab)
    
    # Add special tokens to vocabulary
    for token in special_tokens:
        vocab[next_token_id] = token.encode('utf-8')
        next_token_id += 1
    
    init_time = time.time() - start_time
    profiling_ctx.timings["Vocabulary Initialization"] = init_time
    print(f"⏱️  Vocabulary Initialization: {init_time:.2f}s")
    
    # Check file size to determine training strategy
    with open(input_path, 'rb') as f:
        file_size = f.seek(0, 2)
    
    # For very large files (> 2GB), use memory-efficient trainer
    if BPEConfig.should_use_memory_efficient_trainer(file_size) and strategy == "auto":
        print(f"Very large corpus detected ({BPEConfig.format_size(file_size)}). Using memory-efficient trainer...")
        
        tokenization_start = time.time()
        def create_word_iterator():
            byte_to_id = {v: k for k, v in vocab.items()}
            special_tokens_bytes = {s.encode("utf-8") for s in special_tokens}
            
            # Use streaming strategy for very large files
            token_source = tokenize_parallel_streaming(input_path, special_tokens)
            
            for token in token_source:
                if token in special_tokens_bytes:
                    continue  # Skip special tokens
                
                # Convert token to integer list
                word = [byte_to_id[bytes([b])] for b in token]
                if word:
                    yield word
        
        # Use memory-efficient trainer
        trainer_init_start = time.time()
        trainer = BPETrainerMemoryEfficient(vocab, create_word_iterator)
        trainer_init_time = time.time() - trainer_init_start
        profiling_ctx.timings["Memory-Efficient Trainer Init"] = trainer_init_time
        print(f"⏱️  Memory-Efficient Trainer Init: {trainer_init_time:.2f}s")
    else:
        # For smaller files, use regular in-memory approach
        print(f"Using in-memory trainer for corpus ({BPEConfig.format_size(file_size)})")
        
        # Get pre-tokens using the specified strategy
        tokenization_start = time.time()
        pre_tokens_source = auto_tokenize(input_path, special_tokens, strategy)
        
        # Build word list efficiently
        word_conversion_start = time.time()
        byte_to_id = {v: k for k, v in vocab.items()}
        special_tokens_bytes = {s.encode("utf-8") for s in special_tokens}
        word_list = []
        
        # Process tokens one by one (works for both iterator and list)
        token_count = 0
        for token in pre_tokens_source:
            if token in special_tokens_bytes:
                continue  # Skip special tokens
            
            # Convert token in bytes to integer list based on current vocab
            word = [byte_to_id[bytes([b])] for b in token]
            if word:
                word_list.append(word)
                token_count += 1
        
        tokenization_time = time.time() - tokenization_start
        word_conversion_time = time.time() - word_conversion_start
        profiling_ctx.timings["Tokenization"] = tokenization_time
        profiling_ctx.timings["Word Conversion"] = word_conversion_time
        print(f"⏱️  Tokenization: {tokenization_time:.2f}s")
        print(f"⏱️  Word Conversion: {word_conversion_time:.2f}s")
        print(f"📊 Processed {token_count:,} tokens into {len(word_list):,} words")

        # Use regular in-memory trainer
        trainer_init_start = time.time()
        trainer = BPETrainer(vocab, word_list)
        trainer_init_time = time.time() - trainer_init_start
        profiling_ctx.timings["In-Memory Trainer Init"] = trainer_init_time
        print(f"⏱️  In-Memory Trainer Init: {trainer_init_time:.2f}s")
    # print(word_list)
    merges = []
    
    # Perform merges until we reach desired vocab size
    merge_start = time.time()
    merge_count = 0
    target_merges = vocab_size - len(trainer.vocab)
    print(f"\n🔄 Starting BPE merges: need {target_merges} merges to reach vocab size {vocab_size}")
    
    while len(trainer.vocab) < vocab_size:
        step_start = time.time()
        
        best_pair = trainer.get_best_pair()
        if best_pair is None:
            break
        
        # Record the merge in bytes format
        merges.append((trainer.vocab[best_pair[0]], trainer.vocab[best_pair[1]]))
        
        # Perform the merge
        trainer.merge_pair(best_pair)
        merge_count += 1
        
        step_time = time.time() - step_start
        if merge_count % 10 == 0 or merge_count <= 5:
            print(f"  Merge {merge_count}/{target_merges}: {step_time:.3f}s")
    
    merge_time = time.time() - merge_start
    profiling_ctx.timings["BPE Merges"] = merge_time
    print(f"⏱️  BPE Merges ({merge_count} total): {merge_time:.2f}s")
    print(f"📊 Average merge time: {merge_time/merge_count:.3f}s per merge" if merge_count > 0 else "")
    
    return trainer.vocab, merges


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_with_profiling(use_cprofile=False):
    """Run BPE training with profiling enabled."""
    from pathlib import Path

    # input_path = Path("tests/fixtures/corpus.en") # 129k - Will use sequential processing
    # input_path = Path("data/TinyStoriesV2-GPT4-valid.txt") # 21MB - Will use parallel list strategy
    input_path = Path("data/TinyStoriesV2-GPT4-train.txt") # >2GB - Will use parallel streaming strategy with limited workers
    
    if use_cprofile:
        print("🔍 Running with cProfile detailed profiling...")
        profiler = cProfile.Profile()
        profiler.enable()
        
        vocab, merges = run_train_bpe(
            input_path=input_path, 
            vocab_size=10000,  # As per requirements  
            special_tokens=["<|endoftext|>"],
        )
        
        profiler.disable()
        
        # Print timing summary first
        profiling_ctx.print_summary()
        
        # Then show detailed cProfile results
        print("\n" + "="*50)
        print("DETAILED cProfile RESULTS")
        print("="*50)
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Show top 20 functions
        
        # Save detailed profile to file
        stats.dump_stats('bpe_profile.prof')
        print(f"\n📄 Detailed profile saved to: bpe_profile.prof")
        print("    View with: python -m pstats bpe_profile.prof")
        
    else:
        print("⏱️  Running with timing profiling...")
        vocab, merges = run_train_bpe(
            input_path=input_path, 
            vocab_size=10000,  # As per requirements  
            special_tokens=["<|endoftext|>"],
        )
        
        # Print timing summary
        profiling_ctx.print_summary()
    
    print(f"\n✅ Training complete!")
    print(f"📝 First learned token: {vocab[256]} -> '{vocab[256].decode('utf-8')}'")
    print(f"📝 First 10 merges: {merges[:10]}")
    

if __name__ == "__main__":
    import sys
    
    # Check if user wants cProfile (run with: python train_bpe.py --cprofile)
    use_cprofile = '--cprofile' in sys.argv
    run_with_profiling(use_cprofile)

