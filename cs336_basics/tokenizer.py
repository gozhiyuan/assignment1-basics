import json
from typing import Dict, List, Tuple, Iterable, Iterator, Optional
from pathlib import Path

from .train_bpe import process_chunk, PAT


class BPETokenizer:
    """BPE Tokenizer for encoding text to token IDs and decoding back to text."""
    
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
        """Create tokenizer from vocabulary and merges files.
        
        Args:
            vocab_filepath: Path to vocabulary JSON file
            merges_filepath: Path to merges text file
            special_tokens: Optional list of special tokens
            
        Returns:
            Initialized tokenizer
        """
        # Load vocabulary
        with open(vocab_filepath) as f:
            vocab_data = json.load(f)
            # Convert string keys to integers
            vocab = {int(k): bytes(v) for k, v in vocab_data.items()}
        
        # Load merges
        merges = []
        with open(merges_filepath) as f:
            for line in f:
                first, second = line.strip().split()
                merges.append((first.encode("utf-8"), second.encode("utf-8")))
        
        return cls(vocab, merges, special_tokens)

    def _apply_merges_to_token(self, token: bytes) -> List[bytes]:
        """Apply BPE merges to a single token.
        
        Args:
            token: Bytes to merge
            
        Returns:
            List of merged byte sequences
        """
        # Convert token into list of single bytes
        parts = [bytes([b]) for b in token]
        
        while len(parts) > 1:
            # Find the best merge by rank
            best_rank = float('inf')
            best_idx = None
            
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                if pair in self.merge_ranks:
                    rank = self.merge_ranks[pair]
                    if rank < best_rank:
                        best_rank = rank
                        best_idx = i
            
            if best_idx is None:
                break
                
            # Apply the merge
            merged = parts[best_idx] + parts[best_idx + 1]
            parts = parts[:best_idx] + [merged] + parts[best_idx + 2:]
        
        return parts

    def encode(self, text: str) -> List[int]:
        """Encode text into a sequence of token IDs.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of token IDs
        """
        result = []
        current_text = text
        
        # First handle special tokens (longer ones first)
        while current_text:
            found_special = False
            for special in self.special_tokens:
                if current_text.startswith(special):
                    token_bytes = special.encode("utf-8")
                    result.append(self.byte_to_id[token_bytes])
                    current_text = current_text[len(special):]
                    found_special = True
                    break
            
            if not found_special:
                # Find next special token position
                next_special_pos = len(current_text)
                for special in self.special_tokens:
                    pos = current_text.find(special)
                    if pos != -1:
                        next_special_pos = min(next_special_pos, pos)
                
                # Process text up to next special token
                chunk = current_text[:next_special_pos]
                if chunk:
                    # Use process_chunk for regular text
                    tokens = process_chunk((chunk, []))
                    for token in tokens:
                        # Apply BPE merges
                        merged_parts = self._apply_merges_to_token(token)
                        # Convert to token IDs
                        for part in merged_parts:
                            if part in self.byte_to_id:
                                result.append(self.byte_to_id[part])
                            else:
                                # Handle unknown tokens by splitting into bytes
                                for b in part:
                                    result.append(self.byte_to_id[bytes([b])])
                
                current_text = current_text[next_special_pos:]
        
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Encode an iterable of strings into token IDs.
        
        Args:
            iterable: Iterator yielding strings to encode
            
        Yields:
            Token IDs one at a time
        """
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            
            # Find last complete token boundary
            last_complete = 0
            for match in PAT.finditer(buffer):
                if match.end() < len(buffer):
                    last_complete = match.end()
            
            if last_complete > 0:
                # Encode complete tokens
                for token_id in self.encode(buffer[:last_complete]):
                    yield token_id
                # Keep remainder for next iteration    
                buffer = buffer[last_complete:]
        
        # Encode any remaining text
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
