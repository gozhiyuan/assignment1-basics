
2.4 BPE Tokenizer Training
The BPE tokenizer training procedure consists of three main steps.
Vocabulary initialization The tokenizer vocabulary is a one-to-one mapping from bytestring token to
integer ID. Since we’re training a byte-level BPE tokenizer, our initial vocabulary is simply the set of all
bytes. Since there are 256 possible byte values, our initial vocabulary is of size 256.
Pre-tokenization Once you have a vocabulary, you could, in principle, count how often bytes occur next
to each other in your text and begin merging them starting with the most frequent pair of bytes. However,
this is quite computationally expensive, since we’d have to go take a full pass over the corpus each time
we merge. In addition, directly merging bytes across the corpus may result in tokens that differ only in
punctuation (e.g., dog! vs. dog.). These tokens would get completely different token IDs, even though they
are likely to have high semantic similarity (since they differ only in punctuation).
To avoid this, we pre-tokenize the corpus. You can think of this as a coarse-grained tokenization over the
corpus that helps us count how often pairs of characters appear. For example, the word 'text' might be
a pre-token that appears 10 times. In this case, when we count how often the characters ‘t’ and ‘e’ appear
next to each other, we will see that the word ‘text’ has ‘t’ and ‘e’ adjacent and we can increment their count
by 10 instead of looking through the corpus. Since we’re training a byte-level BPE model, each pre-token is
represented as a sequence of UTF-8 bytes.
The original BPE implementation of Sennrich et al. [2016] pre-tokenizes by simply splitting on whitespace
(i.e., s.split(" ")). In contrast, we’ll use a regex-based pre-tokenizer (used by GPT-2; Radford et al., 2019)
from github.com/openai/tiktoken/pull/234/files:
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
It may be useful to interactively split some text with this pre-tokenizer to get a better sense of its
behavior:
>>> # requires `regex` package
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
6
When using it in your code, however, you should use re.finditer to avoid storing the pre-tokenized words
as you construct your mapping from pre-tokens to their counts.
Compute BPE merges Now that we’ve converted our input text into pre-tokens and represented each
pre-token as a sequence of UTF-8 bytes, we can compute the BPE merges (i.e., train the BPE tokenizer).
At a high level, the BPE algorithm iteratively counts every pair of bytes and identifies the pair with the
highest frequency (“A”, “B”). Every occurrence of this most frequent pair (“A”, “B”) is then merged, i.e.,
replaced with a new token “AB”. This new merged token is added to our vocabulary; as a result, the final
vocabulary after BPE training is the size of the initial vocabulary (256 in our case), plus the number of BPE
merge operations performed during training. For efficiency during BPE training, we do not consider pairs
that cross pre-token boundaries.2 When computing merges, deterministically break ties in pair frequency by
preferring the lexicographically greater pair. For example, if the pairs (“A”, “B”), (“A”, “C”), (“B”, “ZZ”),
and (“BA”, “A”) all have the highest frequency, we’d merge (“BA”, “A”):
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
Special tokens Often, some strings (e.g., <|endoftext|>) are used to encode metadata (e.g., boundaries
between documents). When encoding text, it’s often desirable to treat some strings as “special tokens” that
should never be split into multiple tokens (i.e., will always be preserved as a single token). For example,
the end-of-sequence string <|endoftext|> should always be preserved as a single token (i.e., a single integer
ID), so we know when to stop generating from the language model. These special tokens must be added to
the vocabulary, so they have a corresponding fixed token ID.
Algorithm 1 of Sennrich et al. [2016] contains an inefficient implementation of BPE tokenizer training
(essentially following the steps that we outlined above). As a first exercise, it may be useful to implement
and test this function to test your understanding.

2.5 Experimenting with BPE Tokenizer Training
Let’s train a byte-level BPE tokenizer on the TinyStories dataset. Instructions to find / download the dataset
can be found in Section 1. Before you start, we recommend taking a look at the TinyStories dataset to get
a sense of what’s in the data.
Parallelizing pre-tokenization You will find that a major bottleneck is the pre-tokenization step. You
can speed up pre-tokenization by parallelizing your code with the built-in library multiprocessing. Concretely, we recommend that in parallel implementations of pre-tokenization, you chunk the corpus while
ensuring your chunk boundaries occur at the beginning of a special token. You are free to use the starter
code at the following link verbatim to obtain chunk boundaries, which you can then use to distribute work
across your processes:
https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py
This chunking will always be valid, since we never want to merge across document boundaries. For the
purposes of the assignment, you can always split in this way. Don’t worry about the edge case of receiving
a very large corpus that does not contain <|endoftext|>.
Removing special tokens before pre-tokenization Before running pre-tokenization with the regex
pattern (using re.finditer), you should strip out all special tokens from your corpus (or your chunk, if using
a parallel implementation). Make sure that you split on your special tokens, so that no merging can occur
across the text they delimit. For example, if you have a corpus (or chunk) like [Doc 1]<|endoftext|>[Doc
2], you should split on the special token <|endoftext|>, and pre-tokenize [Doc 1] and [Doc 2] separately,
so that no merging can occur across the document boundary. This can be done using re.split with "|" ⌋ .join(special_tokens) as the delimiter (with careful use of re.escape since | may occur in the special
tokens). The test test_train_bpe_special_tokens will test for this.
Optimizing the merging step The naïve implementation of BPE training in the stylized example above
is slow because for every merge, it iterates over all byte pairs to identify the most frequent pair. However,
the only pair counts that change after each merge are those that overlap with the merged pair. Thus,
BPE training speed can be improved by indexing the counts of all pairs and incrementally updating these
counts, rather than explicitly iterating over each pair of bytes to count pair frequencies. You can get
significant speedups with this caching procedure, though we note that the merging part of BPE training is
not parallelizable in Python.


2.6 BPE Tokenizer: Encoding and Decoding
In the previous part of the assignment, we implemented a function to train a BPE tokenizer on input text
to obtain a tokenizer vocabulary and a list of BPE merges. Now, we will implement a BPE tokenizer that
loads a provided vocabulary and list of merges and uses them to encode and decode text to/from token IDs.
2.6.1 Encoding text
The process of encoding text by BPE mirrors how we train the BPE vocabulary. There are a few major
steps.
Step 1: Pre-tokenize. We first pre-tokenize the sequence and represent each pre-token as a sequence of
UTF-8 bytes, just as we did in BPE training. We will be merging these bytes within each pre-token into
vocabulary elements, handling each pre-token independently (no merges across pre-token boundaries).
Step 2: Apply the merges. We then take the sequence of vocabulary element merges created during BPE
training, and apply it to our pre-tokens in the same order of creation.
10
Example (bpe_encoding): BPE encoding example
For example, suppose our input string is 'the cat ate', our vocabulary is {0: b' ', 1: b'a', 2:
b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b'
at'}, and our learned merges are [(b't', b'h'), (b' ', b'c'), (b' ', 'a'), (b'th', b'e'),
(b' a', b't')]. First, our pre-tokenizer would split this string into ['the', ' cat', ' ate'].
Then, we’ll look at each pre-token and apply the BPE merges.
The first pre-token 'the' is initially represented as [b't', b'h', b'e']. Looking at our list of
merges, we identify the first applicable merge to be (b't', b'h'), and use that to transform the
pre-token into [b'th', b'e']. Then, we go back to the list of merges and identify the next applicable
merge to be (b'th', b'e'), which transforms the pre-token into [b'the']. Finally, looking back at
the list of merges, we see that there are no more that apply to the string (since the entire pre-token
has been merged into a single token), so we are done applying the BPE merges. The corresponding
integer sequence is [9].
Repeating this process for the remaining pre-tokens, we see that the pre-token ' cat' is represented
as [b' c', b'a', b't'] after applying the BPE merges, which becomes the integer sequence [7, 1,
5]. The final pre-token ' ate' is [b' at', b'e'] after applying the BPE merges, which becomes the
integer sequence [10, 3]. Thus, the final result of encoding our input string is [9, 7, 1, 5, 10,
3].
Special tokens. Your tokenizer should be able to properly handle user-defined special tokens when encod￾ing text (provided when constructing the tokenizer).
Memory considerations. Suppose we want to tokenize a large text file that we cannot fit in memory.
To efficiently tokenize this large file (or any other stream of data), we need to break it up into manageable
chunks and process each chunk in-turn, so that the memory complexity is constant as opposed to linear in
the size of the text. In doing so, we need to make sure that a token doesn’t cross chunk boundaries, else
we’ll get a different tokenization than the naïve method of tokenizing the entire sequence in-memory.
2.6.2 Decoding text
To decode a sequence of integer token IDs back to raw text, we can simply look up each ID’s corresponding
entries in the vocabulary (a byte sequence), concatenate them together, and then decode the bytes to a
Unicode string. Note that input IDs are not guaranteed to map to valid Unicode strings (since a user
could input any sequence of integer IDs). In the case that the input token IDs do not produce a valid
Unicode string, you should replace the malformed bytes with the official Unicode replacement character
U+FFFD.3 The errors argument of bytes.decode controls how Unicode decoding errors are handled, and
using errors='replace' will automatically replace malformed data with the replacement marker.
Problem (tokenizer): Implementing the tokenizer (15 points)
Deliverable: Implement a Tokenizer class that, given a vocabulary and a list of merges, encodes
text into integer IDs and decodes integer IDs into text. Your tokenizer should also support user-provided
special tokens (appending them to the vocabulary if they aren’t already there). We recommend the
following interface:
def __init__(self, vocab, merges, special_tokens=None) Construct a tokenizer from a given
vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept
3See en.wikipedia.org/wiki/Specials_(Unicode_block)#Replacement_character for more information about the Unicode
replacement character.
11
the following parameters:
vocab: dict[int, bytes]
merges: list[tuple[bytes, bytes]]
special_tokens: list[str] | None = None
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) Class
method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
(in the same format that your BPE training code output) and (optionally) a list of special
tokens. This method should accept the following additional parameters:
vocab_filepath: str
merges_filepath: str
special_tokens: list[str] | None = None
def encode(self, text: str) -> list[int] Encode an input text into a sequence of token IDs.
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int] Given an iterable of
strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
required for memory-efficient tokenization of large files that we cannot directly load into
memory.
def decode(self, ids: list[int]) -> str Decode a sequence of token IDs into text.
To test your Tokenizer against our provided tests, you will first need to implement the test adapter
at [adapters.get_tokenizer]. Then, run uv run pytest tests/test_tokenizer.py. Your imple￾mentation should be able to pass all tests.




## 2.7 BPE Training Implementation: Step-by-Step Guide

This section provides a detailed walkthrough of the BPE training implementation, explaining the algorithms and data structures used.

### Overview of the Training Process

The BPE training process follows these main steps:

1. **Vocabulary Initialization**: Start with all 256 possible bytes (0-255) plus special tokens
2. **Pre-tokenization**: Split input text using regex patterns while preserving special tokens
3. **Iterative Merging**: Repeatedly find the most frequent byte pair and merge it into a new token
4. **Optimization**: Use efficient data structures to avoid recomputing pair counts from scratch

### The BPETrainer Class: Core Data Structures

The `BPETrainer` class uses several interconnected data structures to efficiently track and update pair frequencies:

```python
class BPETrainer:
    def __init__(self, vocab: Dict[int, bytes], words: List[List[int]]):
        self.vocab = vocab.copy()                    # {token_id: byte_sequence}
        self.words = [word[:] for word in words]     # Deep copy of tokenized words
        self.pair_counts = Counter()                 # {(token1, token2): frequency}
        self.word_pair_indices = defaultdict(list)   # {word_idx: [(pos, pair), ...]}
        self.pair_to_words = defaultdict(set)        # {pair: {word_indices}}
```

#### Example: Data Structure Initialization

Let's trace through a concrete example with the input words: `["hello", "world"]`

**Step 1: Initial State**
```python
# Input words converted to token IDs (assuming ASCII bytes)
words = [
    [104, 101, 108, 108, 111],  # "hello" -> [h, e, l, l, o]
    [119, 111, 114, 108, 100]   # "world" -> [w, o, r, l, d]
]

# Initial vocabulary (first 256 entries are bytes)
vocab = {
    0: b'\x00', 1: b'\x01', ..., 104: b'h', 101: b'e', 108: b'l', 111: b'o', 
    119: b'w', 114: b'r', 100: b'd', ..., 255: b'\xff'
}
```

**Step 2: Build Initial Pair Counts**
```python
# For word 0 "hello" [104, 101, 108, 108, 111]:
pairs_word_0 = [(104,101), (101,108), (108,108), (108,111)]  # (h,e), (e,l), (l,l), (l,o)

# For word 1 "world" [119, 111, 114, 108, 100]:
pairs_word_1 = [(119,111), (111,114), (114,108), (108,100)]  # (w,o), (o,r), (r,l), (l,d)

# Final pair counts:
pair_counts = {
    (104,101): 1,  # (h,e) appears 1 time
    (101,108): 1,  # (e,l) appears 1 time  
    (108,108): 1,  # (l,l) appears 1 time
    (108,111): 1,  # (l,o) appears 1 time
    (119,111): 1,  # (w,o) appears 1 time
    (111,114): 1,  # (o,r) appears 1 time
    (114,108): 1,  # (r,l) appears 1 time
    (108,100): 1   # (l,d) appears 1 time
}
```

**Step 3: Build Auxiliary Data Structures**
```python
# word_pair_indices: For each word, store position and pair info
word_pair_indices = {
    0: [(0, (104,101)), (1, (101,108)), (2, (108,108)), (3, (108,111))],
    1: [(0, (119,111)), (1, (111,114)), (2, (114,108)), (3, (108,100))]
}

# pair_to_words: For each pair, track which words contain it
pair_to_words = {
    (104,101): {0},      # (h,e) is in word 0
    (101,108): {0},      # (e,l) is in word 0
    (108,108): {0},      # (l,l) is in word 0
    (108,111): {0},      # (l,o) is in word 0
    (119,111): {1},      # (w,o) is in word 1
    (111,114): {1},      # (o,r) is in word 1
    (114,108): {1},      # (r,l) is in word 1
    (108,100): {1}       # (l,d) is in word 1
}
```

### The Merging Process: Step-by-Step

#### Finding the Best Pair

The `get_best_pair()` method finds the most frequent pair, using lexicographical order for tie-breaking:

```python
def get_best_pair(self) -> Tuple[int, int]:
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

In our example, all pairs have frequency 1, so lexicographical comparison decides. The pair `(119,111)` representing `(w,o)` would be chosen since 'w' > other first characters.

#### Executing the Merge

When merging pair `(119,111)` (w,o):

**Step 1: Create New Token**
```python
new_token_id = 256  # Next available ID after initial 256 bytes
vocab[256] = vocab[119] + vocab[111]  # b'w' + b'o' = b'wo'
```

**Step 2: Update Affected Words**
```python
# Word 1 changes from [119, 111, 114, 108, 100] to [256, 114, 108, 100]
# This represents "world" -> "wo" + "r" + "l" + "d"

old_word = [119, 111, 114, 108, 100]  # [w, o, r, l, d]
new_word = [256, 114, 108, 100]       # [wo, r, l, d]
```

**Step 3: Update Data Structures**

Remove old pair counts from affected word:
```python
# Remove pairs from word 1: (119,111), (111,114), (114,108), (108,100)
for old_pair in [(119,111), (111,114), (114,108), (108,100)]:
    pair_counts[old_pair] -= 1
    pair_to_words[old_pair].discard(1)
    # Delete entries that reach zero count
```

Add new pair counts:
```python
# New pairs from updated word 1: (256,114), (114,108), (108,100)
new_pairs = [(256,114), (114,108), (108,100)]  # [wo,r], [r,l], [l,d]
for new_pair in new_pairs:
    pair_counts[new_pair] += 1
    pair_to_words[new_pair].add(1)
```

### Why This Approach is Efficient

The key optimization is **incremental updates**:

1. **Avoid Full Rescans**: Instead of recounting all pairs after each merge, we only update counts for affected words
2. **Fast Lookups**: `pair_to_words` allows us to quickly find which words contain a specific pair
3. **Efficient Updates**: `word_pair_indices` helps us efficiently remove old pairs and add new ones

### Complexity Analysis

- **Naive Approach**: O(vocab_size × corpus_size) per merge → O(vocab_size² × corpus_size) total
- **Optimized Approach**: O(affected_words × avg_word_length) per merge → Much faster in practice

The optimized approach is crucial for training on large datasets, as it can be 10-100x faster than the naive implementation.

### Special Token Handling

Special tokens are handled by:
1. **Pre-tokenization**: Split input text around special tokens before applying regex patterns
2. **Vocabulary Isolation**: Add special tokens to vocabulary but exclude them from merge training
3. **Boundary Preservation**: Ensure no merges occur across special token boundaries

```python
# Example: "Hello<|endoftext|>World" becomes:
pre_tokens = [b'Hello', b'<|endoftext|>', b'World']
# Only b'Hello' and b'World' participate in BPE training
# b'<|endoftext|>' remains as a single, indivisible token
```

This ensures special tokens maintain their semantic meaning and are never broken into smaller pieces during tokenization.

### The `encode` Method: From Text to Token IDs

The `encode` method is the core of the tokenizer, converting a string of text into a sequence of token IDs. It follows a multi-step process to handle special tokens and apply BPE merges correctly.

#### Step-by-Step Breakdown:

1. **Sequential Processing**: The method processes the input text sequentially from left to right, ensuring that special tokens are handled correctly, even when they overlap.

2. **Special Token Matching**:
   - It first checks if the text begins with any of the special tokens (longer ones are checked first to handle overlaps).
   - If a match is found, the corresponding token ID is added to the result, and the matched text is removed from the input.

3. **Regular Text Processing**:
   - If no special token is found at the current position, the method processes the text up to the next special token.
   - This chunk of text is then pre-tokenized using the same regex as the BPE trainer.

4. **BPE Merge Application**:
   - For each pre-token, the BPE merge rules are applied iteratively until no more merges are possible.
   - The merged parts are then converted to token IDs and added to the result.

5. **Loop Until Done**: This process repeats until the entire input string has been consumed.

#### Example: `encode("Hello<|endoftext|>World")`

1. **Initial State**: `current_text` = `"Hello<|endoftext|>World"`, `result` = `[]`
2. **First Chunk**:
   - No special token at the start.
   - Next special token is at index 5.
   - Process `"Hello"`.
   - After merging and tokenizing, `result` becomes `[15496]` (assuming GPT-2 merges).
3. **Second Chunk**:
   - `current_text` is now `"<|endoftext|>World"`.
   - It starts with the special token `"<|endoftext|>"`.
   - Token ID for `<|endoftext|>` is added to `result`.
   - `result` is now `[15496, 50256]`.
4. **Third Chunk**:
   - `current_text` is now `"World"`.
   - Process `"World"`.
   - After merging and tokenizing, `result` becomes `[15496, 50256, 1332]`
5. **Final Result**: `[15496, 50256, 1332]`

### Memory-Efficient Encoding with `encode_iterable`

The `encode_iterable` method provides a memory-efficient way to tokenize large files or streams of text without loading the entire content into memory.

#### How It Works:

1. **Buffering**: It reads the input in chunks and appends them to an internal buffer.
2. **Boundary Detection**: It uses the pre-tokenization regex to find the last complete token boundary within the buffer.
3. **Encoding and Yielding**: It encodes the text up to this boundary and yields the resulting token IDs one by one.
4. **Iterative Processing**: The remainder of the buffer is carried over to the next iteration, ensuring that no token boundaries are broken.

This approach is crucial for large-scale tokenization tasks where memory is a concern, as it maintains a constant memory footprint regardless of the input size.
