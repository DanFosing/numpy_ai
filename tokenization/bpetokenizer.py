"""
Byte-Level BPE Tokenizer
--------------------------------
Implementation inspired by Jun Yu Tan's analysis on improving naive BPE:
https://jytan.net/blog/2025/bpe/

Performance:
Uses an "Inverted Index" optimization (O(N) training) and
a "Merge-All" greedy encoding strategy for efficient inference.
"""

import re
import json
from collections import defaultdict, Counter

class BPETokenizer:
    def __init__(self, vocab_size=1000, special_tokens=None):
        self.vocab_size = vocab_size
        if special_tokens is None:
            special_tokens = []
        
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        self.ranks = {}
        
        self.special_tokens = {}
        self.special_ids = {}
        
        curr_id = vocab_size
        for token in special_tokens:
            self.special_tokens[token] = curr_id
            self.special_ids[curr_id] = token
            curr_id += 1
            
        escaped = [re.escape(k) for k in special_tokens]
        spec_pat = "(" + "|".join(escaped) + ")"
        gpt_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+"""
        self.pat = re.compile(spec_pat + "|" + gpt_pat)

    def train(self, text):
        raw_words = []
        for m in self.pat.finditer(text):
            chunk = m.group(0)
            if chunk in self.special_tokens:
                continue
            raw_words.append(list(chunk.encode("utf-8")))

        counts = Counter(tuple(w) for w in raw_words)
        words = list(map(list, counts.keys()))
        freqs = list(counts.values())

        pair_stats = defaultdict(int)
        idx_index = defaultdict(set)

        for i, w in enumerate(words):
            for j in range(len(w) - 1):
                p = (w[j], w[j+1])
                pair_stats[p] += freqs[i]
                idx_index[p].add(i)

        current_id = 256
        rank = 0
        
        while current_id < self.vocab_size and pair_stats:
            best = max(pair_stats, key=pair_stats.get)
            
            self.merges[best] = current_id
            self.ranks[best] = rank
            self.vocab[current_id] = self.vocab[best[0]] + self.vocab[best[1]]

            indices = idx_index.pop(best, ())
            del pair_stats[best]

            for i in indices:
                w = words[i]
                freq = freqs[i]
                
                for j in range(len(w) - 1):
                    p = (w[j], w[j+1])
                    if p in pair_stats:
                        pair_stats[p] -= freq
                        if pair_stats[p] == 0:
                            del pair_stats[p]
                        idx_index[p].discard(i)

                new_w = []
                j = 0
                while j < len(w):
                    if j < len(w)-1 and (w[j], w[j+1]) == best:
                        new_w.append(current_id)
                        j += 2
                    else:
                        new_w.append(w[j])
                        j += 1
                words[i] = new_w

                for j in range(len(new_w) - 1):
                    p = (new_w[j], new_w[j+1])
                    pair_stats[p] += freq
                    idx_index[p].add(i)

            current_id += 1
            rank += 1

    def encode(self, text):
        res = []
        for m in self.pat.finditer(text):
            chunk = m.group(0)
            if chunk in self.special_tokens:
                res.append(self.special_tokens[chunk])
                continue
                
            ids = list(chunk.encode("utf-8"))
            while len(ids) >= 2:
                best_pair = None
                best_rank = None
                for i in range(len(ids)-1):
                    p = (ids[i], ids[i+1])
                    r = self.ranks.get(p)
                    if r is not None:
                        if best_rank is None or r < best_rank:
                            best_rank = r
                            best_pair = p
                
                if best_pair is None:
                    break
                
                new_id = self.merges[best_pair]
                out = []
                i = 0
                while i < len(ids):
                    if i < len(ids)-1 and (ids[i], ids[i+1]) == best_pair:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(ids[i])
                        i += 1
                ids = out
            res.extend(ids)
        return res

    def decode(self, ids):
        parts = []
        for i in ids:
            if i in self.vocab:
                parts.append(self.vocab[i])
            elif i in self.special_ids:
                parts.append(self.special_ids[i].encode("utf-8"))
            else:
                parts.append(b"")
        return b"".join(parts).decode("utf-8", errors="replace")

    def save(self, path):
        merges_export = [
            (p[0], p[1], new_id) 
            for p, new_id in self.merges.items()
        ]
        merges_export.sort(key=lambda x: x[2])
        
        data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "merges": merges_export
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        t = cls(
            vocab_size=data["vocab_size"], 
            special_tokens=list(data["special_tokens"].keys())
        )
        t.special_tokens = data["special_tokens"]
        t.special_ids = {v: k for k, v in t.special_tokens.items()}
        
        for rank, (p0, p1, new_id) in enumerate(data["merges"]):
            pair = (p0, p1)
            t.merges[pair] = new_id
            t.ranks[pair] = rank
            t.vocab[new_id] = t.vocab[p0] + t.vocab[p1]
            
        return t