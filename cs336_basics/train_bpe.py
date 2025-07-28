import os
import regex as re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Any
import multiprocessing as mp
from pathlib import Path
from typing import BinaryIO


# Pre-tokenization regex pattern from GPT-2
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def get_initial_vocab() -> Dict[int, bytes]:
    """Initialize vocabulary with all possible bytes (0-255)."""
    return {i: bytes([i]) for i in range(256)}


def process_chunk(args: tuple[str, List[str]]) -> List[bytes]:
    """Process a single chunk of text.
    Example of input and output:
    Input: "Hello<|endoftext|>World[SEP]Test"
    Output: [b'Hello', b'<|endoftext|>', b'World', b'[SEP]', b'Test']
    
    Args:
        args: Tuple of (chunk_text, special_tokens)
        
    Returns:
        List of pre-tokenized byte sequences
    """
    chunk_text, special_tokens = args

    if not special_tokens:
        return [token.encode("utf-8") for token in PAT.findall(chunk_text)]

    special_pattern = f"({'|'.join(map(re.escape, special_tokens))})"
    special_chunks = re.split(special_pattern, chunk_text)

    pre_tokens = []
    for chunk in special_chunks:
        if not chunk:
            continue
        if chunk in special_tokens:
            pre_tokens.append(chunk.encode("utf-8"))
        else:
            pre_tokens.extend(token.encode("utf-8") for token in PAT.findall(chunk))

    return pre_tokens


def process_chunk_bytes(chunk: bytes, special_tokens: List[str]) -> List[bytes]:
    """Process a chunk of bytes into tokens."""
    # Decode chunk safely
    text = chunk.decode('utf-8', errors='replace')
    return process_chunk((text, special_tokens))


def parallel_pre_tokenize(input_path: str | os.PathLike, special_tokens: List[str]) -> List[bytes]:
    """Pre-tokenize input text in parallel using multiprocessing."""
    # For small files, skip multiprocessing overhead
    with open(input_path, 'rb') as f:
        file_size = f.seek(0, 2)  # Get file size
        f.seek(0)  # Reset to beginning
        
        if file_size < 1024 * 1024:  # Less than 1MB, process sequentially
            chunk = f.read()
            return process_chunk_bytes(chunk, special_tokens)
    
    # For larger files, use streaming with multiprocessing
    n_cores = max(1, mp.cpu_count() - 1)  # Leave one core for system
    
    # Get chunk boundaries that respect special tokens
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, desired_num_chunks=n_cores, split_special_token=special_tokens[0].encode("utf-8")
        )
    
    # Read chunks based on boundaries
    chunks = []
    with open(input_path, 'rb') as f:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start))
    
    # Process chunks in parallel
    with mp.Pool(n_cores) as pool:
        results = pool.starmap(process_chunk_bytes, [(chunk, special_tokens) for chunk in chunks])
    
    # Combine results
    return [token for chunk_tokens in results for token in chunk_tokens]


class BPETrainer:
    """Optimized BPE trainer with incremental updates."""
    
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
        # print(word_idx, word)
        # print(pair, self.pair_counts[pair])
        # print(self.word_pair_indices[0])

    
    def get_best_pair(self) -> Tuple[int, int]:
        """Find the most frequent pair with lexicographical tie-breaking.

        Returns:
            Tuple[int, int]: The most frequent pair with lexicographical tie-breaking.
        """
        if not self.pair_counts:
            return None
        
        # Sort pairs: primary key frequency (desc), secondary key pair bytes (desc)
        # This correctly handles tie-breaking as per the spec.
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
        """
        new_token_id = len(self.vocab)
        self.vocab[new_token_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
        
        # Get all words containing this pair
        affected_words = list(self.pair_to_words[pair])
        
        for word_idx in affected_words:
            old_word = self.words[word_idx]
            new_word = []
            
            # Remove old pair counts for this word
            for pos, old_pair in self.word_pair_indices[word_idx]:
                self.pair_counts[old_pair] -= 1
                self.pair_to_words[old_pair].discard(word_idx)
                if self.pair_counts[old_pair] == 0:
                    del self.pair_counts[old_pair]
                    del self.pair_to_words[old_pair]
            
            # Apply merge to create new word
            i = 0
            while i < len(old_word):
                if i < len(old_word) - 1 and old_word[i] == pair[0] and old_word[i + 1] == pair[1]:
                    new_word.append(new_token_id)
                    i += 2
                else:
                    new_word.append(old_word[i])
                    i += 1
            
            # Update word and add new pair counts
            self.words[word_idx] = new_word
            self.word_pair_indices[word_idx] = []
            
            for i in range(len(new_word) - 1):
                new_pair = (new_word[i], new_word[i + 1])
                self.pair_counts[new_pair] += 1
                self.word_pair_indices[word_idx].append((i, new_pair))
                self.pair_to_words[new_pair].add(word_idx)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab: The trained tokenizer vocabulary
            merges: BPE merges ordered by creation
    """
    # Initialize vocabulary with bytes 0-255
    vocab = get_initial_vocab()
    next_token_id = len(vocab)
    
    # Add special tokens to vocabulary
    for token in special_tokens:
        vocab[next_token_id] = token.encode('utf-8')
        next_token_id += 1
    
    # Pre-tokenize input text efficiently
    pre_tokens = parallel_pre_tokenize(input_path, special_tokens)
    
    # Convert tokens to integer lists based on current vocab
    byte_to_id = {v: k for k, v in vocab.items()}
    special_tokens_bytes = {s.encode("utf-8") for s in special_tokens}
    word_list = []
    for token in pre_tokens:
        if token in special_tokens_bytes:
            continue
        word = [byte_to_id[bytes([b])] for b in token]
        if word:
            word_list.append(word)

    # Initialize BPE trainer
    trainer = BPETrainer(vocab, word_list)
    merges = []
    
    # Perform merges until we reach desired vocab size
    while len(trainer.vocab) < vocab_size:
        best_pair = trainer.get_best_pair()
        if best_pair is None:
            break
        
        # Record the merge in bytes format
        merges.append((trainer.vocab[best_pair[0]], trainer.vocab[best_pair[1]]))
        
        # Perform the merge
        trainer.merge_pair(best_pair)
    
    return trainer.vocab, merges

if __name__ == "__main__":

    from pathlib import Path

    input_path = Path("tests/fixtures/corpus.en")
    vocab, merges = run_train_bpe(
        input_path=input_path, 
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    print(vocab[256], vocab[256].decode('utf-8'))
    print(merges[:10])

