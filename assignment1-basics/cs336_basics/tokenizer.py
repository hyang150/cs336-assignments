import regex as re
import json
import os
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Union
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

# GPT-4 / GPT-2 standard split pattern
# This regex splits text into: contractions, letters, numbers, or anything else (punctuation)
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class BPETokenizer:
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[bytes, bytes], int] = {}
        self.vocab: Dict[int, bytes] = {}
        self.special_tokens: Dict[str, int] = {}
        self.pattern = PAT

    def _get_bytes_range(self, special_tokens: List[str]):
        """Initialize basic vocab (0-255) + special tokens."""
        # 0-255 bytes
        vocab = {i: bytes([i]) for i in range(256)}
        
        # Add special tokens starting from 256
        for i, token in enumerate(special_tokens):
            token_id = 256 + i
            vocab[token_id] = token.encode("utf-8")
            self.special_tokens[token] = token_id
            
        return vocab

    @staticmethod
    def _word2bytes(word: str) -> Tuple[bytes, ...]:
        """Convert word string to tuple of bytes (immutable for dict keys)."""
        # Convert string to utf-8 bytes, then to a tuple of single-byte objects
        # e.g. "Hi" -> (b'H', b'i')
        return tuple(bytes([b]) for b in word.encode('utf-8'))

    @staticmethod
    def _count_word_freq(text: str) -> Dict[Tuple[bytes, ...], int]:
        """Split text using regex and count byte-tuple frequencies."""
        word_cnt = defaultdict(int)
        for m in PAT.finditer(text):
            word = m.group(0)
            word_bytes = BPETokenizer._word2bytes(word)
            if len(word_bytes) > 0:
                word_cnt[word_bytes] += 1
        return word_cnt

    @staticmethod
    def _apply_merge(word_bytes: Tuple[bytes, ...], pair: Tuple[bytes, bytes]) -> Tuple[bytes, ...]:
        """Apply a specific merge to a word's byte tuple."""
        # This creates a new tuple where consecutive occurrences of pair are merged
        new_word_bytes = []
        i = 0
        merged_token = pair[0] + pair[1]
        
        while i < len(word_bytes):
            if i < len(word_bytes) - 1 and word_bytes[i] == pair[0] and word_bytes[i+1] == pair[1]:
                new_word_bytes.append(merged_token)
                i += 2
            else:
                new_word_bytes.append(word_bytes[i])
                i += 1
        return tuple(new_word_bytes)

    def _update_stats(self, word_cnt, pair_cnt, merge_pair):
        """
        Incremental update of statistics.
        Instead of re-scanning text, we mathematically update the counts.
        Efficient O(1) logic from the original optimized code.
        """
        new_word_cnt = defaultdict(int)
        new_pair_cnt = pair_cnt.copy()

        for word_bytes, cnt in word_cnt.items():
            # If the word doesn't contain the first part of the pair, it can't be merged
            # Optimization: check if first byte of pair is in word
            if merge_pair[0] not in word_bytes:
                new_word_cnt[word_bytes] += cnt
                continue

            # Calculate pairs in the OLD word to remove them
            old_pairs = list(zip(word_bytes[:-1], word_bytes[1:]))
            if merge_pair not in old_pairs:
                new_word_cnt[word_bytes] += cnt
                continue

            # Apply merge to get NEW word
            new_word = self._apply_merge(word_bytes, merge_pair)
            new_word_cnt[new_word] += cnt

            # 1. Remove counts of old pairs
            for p in old_pairs:
                new_pair_cnt[p] -= cnt
                if new_pair_cnt[p] == 0:
                    del new_pair_cnt[p]

            # 2. Add counts of new pairs
            new_pairs = list(zip(new_word[:-1], new_word[1:]))
            for p in new_pairs:
                new_pair_cnt[p] += cnt

        return new_word_cnt, new_pair_cnt

    def train(self, input_path: Union[str, os.PathLike], special_tokens: List[str] = None):
        """Train the BPE tokenizer on a file."""
        special_tokens = special_tokens or []
        # Sort special tokens by length (desc) to ensure greedy matching
        special_tokens = sorted(special_tokens, key=len, reverse=True)
        
        # print(f"Reading {input_path}...")
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        # 1. Split logic handling special tokens
        chunks = [text]
        if special_tokens:
            pattern_str = "|".join(re.escape(tok) for tok in special_tokens)
            pattern = re.compile(f"({pattern_str})")
            raw_chunks = pattern.split(text)
            # Filter out empty strings and the special tokens themselves (we don't train BPE on special tokens)
            chunks = [c for c in raw_chunks if c and c not in special_tokens]

        # 2. Parallel Count Words
        # print("Counting words...")
        # Use simple map if very few chunks to avoid process overhead on small files
        if len(chunks) < 4:
            word_dicts = list(map(self._count_word_freq, chunks))
        else:
            word_dicts = process_map(self._count_word_freq, chunks, chunksize=1, max_workers=os.cpu_count(), disable=True)

        # Merge results from parallel processing
        word_cnt = defaultdict(int)
        for d in word_dicts:
            for k, v in d.items():
                word_cnt[k] += v

        # 3. Initial Pair Statistics
        pair_cnt = defaultdict(int)
        for word_bytes, cnt in word_cnt.items():
            for pair in zip(word_bytes[:-1], word_bytes[1:]):
                pair_cnt[pair] += cnt

        # 4. Initialize Vocab
        self.vocab = self._get_bytes_range(special_tokens)
        base_vocab_size = len(self.vocab)
        num_merges = self.vocab_size - base_vocab_size

        # 5. Training Loop
        # print(f"Training for {num_merges} merges...")
        # Disable tqdm for tests to keep output clean, enable if running manually
        iterator = range(num_merges)
        
        for i in iterator:
            if not pair_cnt:
                break

            # EDGE CASE: Deterministic Tie-Breaking
            # Sort by frequency (desc), then by pair bytes (lexicographical asc)
            # This is critical for passing tests that check for exact reproduction
            best_pair = max(pair_cnt.items(), key=lambda x: (x[1], x[0]))[0]

            # Register Merge
            new_token_bytes = best_pair[0] + best_pair[1]
            token_id = base_vocab_size + i
            self.vocab[token_id] = new_token_bytes
            self.merges[best_pair] = token_id

            # Efficient Update
            word_cnt, pair_cnt = self._update_stats(word_cnt, pair_cnt, best_pair)

        return self.vocab, self.merges

    def encode(self, text: str) -> List[int]:
        """Encodes text into a list of token IDs."""
        if not self.vocab:
            raise ValueError("Tokenizer not trained yet!")

        ids = []
        # Use the regex to split into words
        for m in self.pattern.finditer(text):
            word = m.group(0)
            
            # Start with bytes
            word_bytes = self._word2bytes(word)
            
            # Apply merges iteratively
            while len(word_bytes) >= 2:
                stats = {}
                for pair in zip(word_bytes[:-1], word_bytes[1:]):
                    stats[pair] = self.merges.get(pair, float('inf'))
                
                # Find the pair with the lowest merge index (earliest learned merge)
                best_pair = min(stats, key=lambda p: stats.get(p, float('inf')))
                
                if best_pair not in self.merges:
                    break # No more applicable merges
                
                word_bytes = self._apply_merge(word_bytes, best_pair)

            # Map final byte chunks to IDs
            reverse_vocab = {v: k for k, v in self.vocab.items()}
            
            for part in word_bytes:
                if part in reverse_vocab:
                    ids.append(reverse_vocab[part])
                else:
                    for b in part:
                        ids.append(b)
                        
        return ids

    def save(self, prefix: str):
        """Save vocab and merges."""
        vocab_export = {id: b.decode('latin-1') for id, b in self.vocab.items()}
        with open(f"{prefix}.vocab", "w") as f:
            json.dump(vocab_export, f)
            
        merges_export = []
        for (p1, p2), idx in self.merges.items():
            merges_export.append({
                "p1": p1.decode('latin-1'),
                "p2": p2.decode('latin-1'),
                "id": idx
            })
        with open(f"{prefix}.merges", "w") as f:
            json.dump(merges_export, f)

# --- ADAPTER FOR ASSIGNMENT TESTS ---
def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    Wrapper function to match the signature expected by cs336_basics tests.
    """
    # 1. Instantiate the class
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    
    # 2. Run training
    vocab, merges_dict = tokenizer.train(input_path, special_tokens=special_tokens)
    
    # 3. Format output
    # The class returns a dict for merges, but the test expects a list of tuples 
    # sorted by creation order (which dict keys preserve in Python 3.7+)
    merges_list = list(merges_dict.keys())
    
    return vocab, merges_list

if __name__ == "__main__":
    # Dummy file creation for testing
    with open("test_corpus.txt", "w", encoding="utf-8") as f:
        f.write("Hello world! Hello tokenizer. This is a test. 世界你好。")

    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train("test_corpus.txt", special_tokens=["<|endoftext|>"])
    
    # Test Encoding
    ids = tokenizer.encode("Hello world! 世界")
    print(f"Encoded IDs: {ids}")
    
    # Verify correctness (basic check)
    decoded_bytes = b"".join([tokenizer.vocab[i] for i in ids])
    print(f"Decoded: {decoded_bytes.decode('utf-8')}")