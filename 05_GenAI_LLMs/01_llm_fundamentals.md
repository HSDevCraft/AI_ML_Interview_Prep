# LLM Fundamentals - Complete Guide

## Table of Contents
1. [Architecture Evolution](#architecture-evolution)
2. [Decoder-Only Architectures](#decoder-only-architectures)
3. [Tokenization Deep Dive](#tokenization-deep-dive)
4. [Pre-training Objectives](#pre-training-objectives)
5. [Scaling Laws](#scaling-laws)
6. [Modern LLM Components](#modern-llm-components)
7. [Interview Questions](#interview-questions)

---

## 1. Architecture Evolution

### Timeline of Key Models

```
Pre-Transformer Era:
- 2013: Word2Vec (word embeddings)
- 2014: Seq2Seq with Attention
- 2015: Attention mechanisms refined

Transformer Era:
- 2017: Transformer ("Attention Is All You Need")
- 2018: GPT-1 (117M), BERT (340M), ELMo
- 2019: GPT-2 (1.5B), T5, XLNet, RoBERTa
- 2020: GPT-3 (175B), mT5, BART
- 2021: Codex, FLAN, Jurassic-1

ChatGPT Era:
- 2022: ChatGPT, InstructGPT, Chinchilla, PaLM
- 2023: GPT-4, LLaMA 1&2, Claude 1&2, Mistral, Mixtral
- 2024: LLaMA 3, Claude 3, Gemini 1.5, GPT-4o, Qwen 2
```

### Architecture Taxonomy

```python
"""
Three main architectures for language models:

1. Encoder-Only (BERT, RoBERTa)
   - Bidirectional attention
   - Pre-training: MLM, NSP
   - Use: Classification, NER, embeddings
   - Cannot generate text naturally

2. Decoder-Only (GPT, LLaMA, Mistral)
   - Causal (left-to-right) attention
   - Pre-training: Next token prediction
   - Use: Text generation, few-shot learning
   - Most scalable, dominant architecture

3. Encoder-Decoder (T5, BART)
   - Encoder: bidirectional
   - Decoder: causal + cross-attention
   - Pre-training: Span corruption, denoising
   - Use: Translation, summarization
"""

# Model comparison
MODEL_SPECS = {
    # Encoder-only
    "bert-base": {"type": "encoder", "params": "110M", "context": 512},
    "bert-large": {"type": "encoder", "params": "340M", "context": 512},
    "roberta-large": {"type": "encoder", "params": "355M", "context": 512},
    
    # Decoder-only
    "gpt-2": {"type": "decoder", "params": "1.5B", "context": 1024},
    "gpt-3": {"type": "decoder", "params": "175B", "context": 4096},
    "gpt-4": {"type": "decoder", "params": "~1.7T MoE", "context": "128K"},
    "llama-2-7b": {"type": "decoder", "params": "7B", "context": 4096},
    "llama-2-70b": {"type": "decoder", "params": "70B", "context": 4096},
    "llama-3-70b": {"type": "decoder", "params": "70B", "context": "128K"},
    "mistral-7b": {"type": "decoder", "params": "7B", "context": 32768},
    "mixtral-8x7b": {"type": "decoder", "params": "46.7B MoE", "context": 32768},
    
    # Encoder-decoder
    "t5-base": {"type": "enc-dec", "params": "220M", "context": 512},
    "t5-11b": {"type": "enc-dec", "params": "11B", "context": 512},
    "flan-t5-xxl": {"type": "enc-dec", "params": "11B", "context": 512},
}
```

---

## 2. Decoder-Only Architectures

### Modern GPT Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Used in LLaMA, Mistral instead of LayerNorm.
    
    Simpler and faster than LayerNorm:
    - No mean subtraction
    - No learned bias
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Used in LLaMA, Mistral, GPT-NeoX.
    
    Advantages:
    - Relative position encoded in attention
    - Better extrapolation to longer sequences
    - Applied to Q, K only
    """
    def __init__(self, dim, max_seq_len=4096, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos/sin
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x, seq_len):
        return (
            self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0),
            self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        )


def rotate_half(x):
    """Rotate half the hidden dims."""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary embeddings to queries and keys."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).
    Used in LLaMA 2, Mistral.
    
    KV heads < Q heads, reducing memory/compute.
    
    Variants:
    - MHA: num_kv_heads = num_heads (standard)
    - GQA: num_kv_heads < num_heads (grouped)
    - MQA: num_kv_heads = 1 (multi-query)
    """
    def __init__(self, d_model, num_heads, num_kv_heads, max_seq_len=4096):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)
    
    def forward(self, x, attention_mask=None, past_kv=None):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Transpose for attention: (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # KV cache for generation
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Repeat K, V for grouped query attention
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Causal mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, k.size(2), device=x.device),
                diagonal=k.size(2) - seq_len + 1
            ).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        
        return output, (k, v)


class SwiGLU(nn.Module):
    """
    SwiGLU activation (used in LLaMA, PaLM).
    
    SwiGLU(x) = Swish(xW_1) ⊙ (xW_2)
    
    Better than ReLU/GELU in practice.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # Gate
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LLaMABlock(nn.Module):
    """
    LLaMA-style transformer block.
    
    Key differences from GPT-2:
    - RMSNorm instead of LayerNorm
    - SwiGLU instead of GELU
    - RoPE instead of learned positional embeddings
    - GQA instead of MHA (LLaMA 2)
    - Pre-normalization
    """
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff):
        super().__init__()
        self.attention_norm = RMSNorm(d_model)
        self.attention = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.feed_forward = SwiGLU(d_model, d_ff)
    
    def forward(self, x, attention_mask=None, past_kv=None):
        # Pre-norm attention with residual
        h = x + self.attention(self.attention_norm(x), attention_mask, past_kv)[0]
        # Pre-norm FFN with residual
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class LLaMA(nn.Module):
    """
    Complete LLaMA-style model.
    
    LLaMA 2 7B config:
    - d_model: 4096
    - num_heads: 32
    - num_kv_heads: 32 (8 for 70B)
    - num_layers: 32
    - d_ff: 11008
    - vocab_size: 32000
    """
    def __init__(self, vocab_size, d_model, num_heads, num_kv_heads, 
                 num_layers, d_ff, max_seq_len=4096):
        super().__init__()
        
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            LLaMABlock(d_model, num_heads, num_kv_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight
    
    def forward(self, input_ids, attention_mask=None):
        h = self.embed_tokens(input_ids)
        
        for layer in self.layers:
            h = layer(h, attention_mask)
        
        h = self.norm(h)
        logits = self.lm_head(h)
        
        return logits
```

---

## 3. Tokenization Deep Dive

### BPE Implementation

```python
import regex as re
from collections import defaultdict

class BPETokenizer:
    """
    Byte Pair Encoding tokenizer.
    
    Algorithm:
    1. Start with character-level vocabulary
    2. Count adjacent pair frequencies
    3. Merge most frequent pair
    4. Repeat until vocab_size reached
    
    Used by: GPT-2, GPT-3, GPT-4
    """
    
    def __init__(self, vocab_size=50257):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.pattern = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
        )
    
    def get_stats(self, ids):
        """Count frequency of adjacent pairs."""
        counts = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts
    
    def merge(self, ids, pair, new_id):
        """Merge all occurrences of pair."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def train(self, text):
        """Train BPE on text corpus."""
        # Pre-tokenize with regex
        words = re.findall(self.pattern, text)
        
        # Convert to byte sequences
        word_freqs = defaultdict(int)
        for word in words:
            word_freqs[tuple(word.encode('utf-8'))] += 1
        
        # Initialize vocabulary with bytes
        self.vocab = {i: bytes([i]) for i in range(256)}
        
        num_merges = self.vocab_size - 256
        
        for i in range(num_merges):
            # Count pairs across all words
            pair_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                for pair in zip(word, word[1:]):
                    pair_freqs[pair] += freq
            
            if not pair_freqs:
                break
            
            # Find most frequent pair
            best_pair = max(pair_freqs, key=pair_freqs.get)
            new_id = 256 + i
            
            # Merge in all words
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = tuple(self.merge(list(word), best_pair, new_id))
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs
            
            # Update merges and vocab
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            if (i + 1) % 1000 == 0:
                print(f"Merge {i+1}/{num_merges}")
    
    def encode(self, text):
        """Encode text to token IDs."""
        words = re.findall(self.pattern, text)
        ids = []
        
        for word in words:
            word_ids = list(word.encode('utf-8'))
            
            while len(word_ids) >= 2:
                pairs = [(word_ids[i], word_ids[i+1]) for i in range(len(word_ids)-1)]
                # Find pair with lowest merge index
                mergeable = [(p, self.merges[p]) for p in pairs if p in self.merges]
                if not mergeable:
                    break
                best_pair = min(mergeable, key=lambda x: x[1])[0]
                word_ids = self.merge(word_ids, best_pair, self.merges[best_pair])
            
            ids.extend(word_ids)
        
        return ids
    
    def decode(self, ids):
        """Decode token IDs to text."""
        bytes_list = b''.join(self.vocab[i] for i in ids)
        return bytes_list.decode('utf-8', errors='replace')


# SentencePiece (used by LLaMA, T5)
"""
Key differences from BPE:
- Treats text as raw character sequence (no pre-tokenization)
- Uses unigram language model or BPE
- Includes special tokens: <s>, </s>, <unk>, <pad>
- Better for multilingual models
"""

# Using HuggingFace tokenizers
from transformers import AutoTokenizer

# GPT-2 tokenizer (BPE)
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = gpt2_tokenizer.encode("Hello, world!")
