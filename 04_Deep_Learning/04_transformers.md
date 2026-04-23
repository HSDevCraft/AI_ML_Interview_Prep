# Transformers - Complete Guide

## Table of Contents
1. [Self-Attention Mechanism](#self-attention-mechanism)
2. [Transformer Architecture](#transformer-architecture)
3. [Positional Encoding](#positional-encoding)
4. [Encoder-Decoder Transformers](#encoder-decoder-transformers)
5. [Pre-trained Models](#pre-trained-models)
6. [Efficient Transformers](#efficient-transformers)
7. [Interview Questions](#interview-questions)

---

## 1. Self-Attention Mechanism

### Scaled Dot-Product Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    Attention(Q, K, V) = softmax(QK^T / √d_k) V
    
    Args:
        query: (batch, num_heads, seq_len, d_k)
        key: (batch, num_heads, seq_len, d_k)
        value: (batch, num_heads, seq_len, d_v)
        mask: Optional mask for padding/causal attention
    
    Why scale by √d_k?
    - Dot products grow with d_k, pushing softmax into saturation
    - Scaling keeps variance stable
    """
    d_k = query.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: (batch, num_heads, seq_len, seq_len)
    
    # Apply mask (for padding or causal attention)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply dropout
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # Weighted sum of values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


# Self-attention vs Cross-attention
"""
Self-attention: Q, K, V all come from same sequence
- Used in encoder to attend to other positions
- Used in decoder (masked) to attend to previous positions

Cross-attention: Q from decoder, K, V from encoder
- Used in decoder to attend to encoder outputs
- Connects encoder and decoder in seq2seq
"""
```

### Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention allows the model to jointly attend
    to information from different representation subspaces.
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    
    Why multiple heads?
    - Different heads can learn different patterns
    - One head might focus on syntax, another on semantics
    - Increases model capacity without increasing depth
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Reshape to (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_output, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask, self.dropout
        )
        
        # Concatenate heads: (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        return output, attn_weights


# Efficient multi-head attention (single matrix multiplication)
class EfficientMultiHeadAttention(nn.Module):
    """Compute Q, K, V projections in single operation."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Combined projection for Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Single projection
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, d_k)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask, self.dropout)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(attn_output)
```

---

## 2. Transformer Architecture

### Feed-Forward Network

```python
class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    
    Or with GELU (modern transformers):
    FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
    
    Typically d_ff = 4 * d_model
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.SiLU()  # Swish
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


# GLU variants (used in modern LLMs)
class GatedFeedForward(nn.Module):
    """
    Gated Linear Unit variant (SwiGLU, GeGLU).
    
    Used in LLaMA, PaLM, etc.
    FFN(x) = (xW_1 ⊙ σ(xW_gate)) W_2
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(
            F.silu(self.gate(x)) * self.linear1(x)
        ))
```

### Transformer Encoder Layer

```python
class TransformerEncoderLayer(nn.Module):
    """
    Single encoder layer:
    
    x → LayerNorm → MultiHeadAttn → Dropout → (+x) → 
    → LayerNorm → FFN → Dropout → (+) → output
    
    Pre-LN (shown above) vs Post-LN:
    - Pre-LN: More stable training, used in most modern models
    - Post-LN: Original Transformer, needs careful LR warmup
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, pre_norm=True):
        super().__init__()
        self.pre_norm = pre_norm
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        if self.pre_norm:
            # Pre-LayerNorm (modern)
            attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)
            x = x + self.dropout1(attn_out)
            
            ff_out = self.feed_forward(self.norm2(x))
            x = x + self.dropout2(ff_out)
        else:
            # Post-LayerNorm (original)
            attn_out, _ = self.self_attn(x, x, x, mask)
            x = self.norm1(x + self.dropout1(attn_out))
            
            ff_out = self.feed_forward(x)
            x = self.norm2(x + self.dropout2(ff_out))
        
        return x


class TransformerEncoder(nn.Module):
    """Stack of encoder layers."""
    
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```

### Transformer Decoder Layer

```python
class TransformerDecoderLayer(nn.Module):
    """
    Single decoder layer:
    
    x → LayerNorm → Masked Self-Attn → (+x) →
    → LayerNorm → Cross-Attn (with encoder) → (+) →
    → LayerNorm → FFN → (+) → output
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Masked self-attention (causal)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # Cross-attention to encoder
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # Feed-forward
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, tgt_mask)
        x = x + self.dropout(attn_out)
        
        # Cross-attention
        x_norm = self.norm2(x)
        attn_out, _ = self.cross_attn(x_norm, encoder_output, encoder_output, src_mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward
        x_norm = self.norm3(x)
        ff_out = self.feed_forward(x_norm)
        x = x + self.dropout(ff_out)
        
        return x
```

---

## 3. Positional Encoding

### Sinusoidal Positional Encoding

```python
class SinusoidalPositionalEncoding(nn.Module):
    """
    Original positional encoding from "Attention Is All You Need".
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Properties:
    - Fixed, not learned
    - Can extrapolate to longer sequences
    - PE(pos+k) can be represented as linear function of PE(pos)
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

### Learned Positional Embedding

```python
class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional embeddings (BERT, GPT-2).
    
    Pros: Can learn task-specific patterns
    Cons: Cannot extrapolate beyond max_len
    """
    
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return x + self.embedding(positions)
```

### Rotary Positional Embedding (RoPE)

```python
class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) - used in LLaMA, GPT-NeoX, etc.
    
    Key idea: Encode position as rotation in complex plane.
    - Relative position encoded in attention score
    - Can extrapolate to longer sequences
    - Applied to Q and K, not added to embeddings
    
    q_rot = q * cos(θ) + rotate_half(q) * sin(θ)
    """
    
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos/sin
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer('cos', emb.cos())
        self.register_buffer('sin', emb.sin())
    
    def rotate_half(self, x):
        """Rotate half the hidden dims."""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, q, k, seq_len):
        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)
        
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_embed, k_embed
```

### ALiBi (Attention with Linear Biases)

```python
class ALiBiPositionalBias(nn.Module):
    """
    ALiBi: No positional encoding in embeddings.
    Instead, add linear bias to attention scores.
    
    bias(i, j) = -m * |i - j|
    
    where m is a head-specific slope.
    
    Used in BLOOM, MPT.
    Advantages: Better length extrapolation than RoPE.
    """
    
    def __init__(self, num_heads, max_len=2048):
        super().__init__()
        
        # Compute slopes for each head
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', torch.tensor(slopes))
        
        # Precompute bias matrix
        positions = torch.arange(max_len)
        bias = -torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
        self.register_buffer('bias', bias)
    
    def _get_slopes(self, num_heads):
        """Get slopes using geometric sequence."""
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            closest_power = 2 ** math.floor(math.log2(num_heads))
            return (
                get_slopes_power_of_2(closest_power) +
                get_slopes_power_of_2(2 * closest_power)[0::2][:num_heads - closest_power]
            )
    
    def forward(self, seq_len):
        # Returns (num_heads, seq_len, seq_len)
        return self.slopes.unsqueeze(1).unsqueeze(1) * self.bias[:seq_len, :seq_len]
```

---

## 4. Encoder-Decoder Transformers

### Full Transformer (Translation)

```python
class Transformer(nn.Module):
    """
    Full encoder-decoder Transformer for sequence-to-sequence tasks.
    """
    
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        
        # Encoder
        self.encoder = TransformerEncoder(
            d_model, num_heads, d_ff, num_encoder_layers, dropout
        )
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        
        self.d_model = d_model
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def encode(self, src, src_mask=None):
        src = self.src_embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        
        for layer in self.decoder_layers:
            tgt = layer(tgt, encoder_output, src_mask, tgt_mask)
        
        return self.decoder_norm(tgt)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return self.output_proj(decoder_output)
```

---

## 5. Pre-trained Models

### BERT Architecture

```python
class BERTEmbedding(nn.Module):
    """
    BERT Embedding = Token + Segment + Position
    """
    
    def __init__(self, vocab_size, d_model, max_len=512, num_segments=2, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.segment_embedding = nn.Embedding(num_segments, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids, segment_ids):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device)
        
        embeddings = (
            self.token_embedding(input_ids) +
            self.segment_embedding(segment_ids) +
            self.position_embedding(position_ids)
        )
        
        return self.dropout(self.norm(embeddings))


class BERT(nn.Module):
    """
    BERT: Bidirectional Encoder Representations from Transformers
    
    Pre-training objectives:
    1. Masked Language Modeling (MLM): Predict [MASK] tokens
    2. Next Sentence Prediction (NSP): Is sentence B after A?
    
    Architecture: Encoder-only Transformer
    """
    
    def __init__(self, vocab_size, d_model=768, num_heads=12, 
                 num_layers=12, d_ff=3072, dropout=0.1):
        super().__init__()
        
        self.embedding = BERTEmbedding(vocab_size, d_model)
        self.encoder = TransformerEncoder(d_model, num_heads, d_ff, num_layers, dropout)
        
        # MLM head
        self.mlm_head = nn.Linear(d_model, vocab_size)
        
        # NSP head (uses [CLS] token)
        self.nsp_head = nn.Linear(d_model, 2)
    
    def forward(self, input_ids, segment_ids, attention_mask=None):
        embeddings = self.embedding(input_ids, segment_ids)
        encoder_output = self.encoder(embeddings, attention_mask)
        
        mlm_logits = self.mlm_head(encoder_output)
        nsp_logits = self.nsp_head(encoder_output[:, 0])  # [CLS] token
        
        return mlm_logits, nsp_logits


# BERT variants comparison
"""
BERT-Base: 12 layers, 768 hidden, 12 heads, 110M params
BERT-Large: 24 layers, 1024 hidden, 16 heads, 340M params

RoBERTa: BERT with better training
- No NSP, dynamic masking, more data, longer training

ALBERT: Lite BERT
- Factorized embedding, cross-layer parameter sharing

DistilBERT: Distilled BERT
- 6 layers, 40% smaller, 60% faster, 97% performance
"""
```

### GPT Architecture

```python
class GPT(nn.Module):
    """
    GPT: Generative Pre-trained Transformer
    
    Architecture: Decoder-only Transformer
    Pre-training: Autoregressive language modeling
    
    P(sequence) = ∏ P(token_i | token_1, ..., token_{i-1})
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,
        max_len=1024,
        dropout=0.1
    ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Decoder layers (no cross-attention)
        self.layers = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying: output embedding = input embedding
        self.head.weight = self.token_embedding.weight
    
    def forward(self, input_ids, attention_mask=None):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device)
        
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        x = self.dropout(x)
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device),
            diagonal=1
        ).bool()
        
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        x = self.norm(x)
        logits = self.head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=100, temperature=1.0, top_k=50):
        """Autoregressive generation."""
        self.eval()
        
        for _ in range(max_new_tokens):
            logits = self.forward(prompt_ids)[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            prompt_ids = torch.cat([prompt_ids, next_token], dim=1)
        
        return prompt_ids


class GPTBlock(nn.Module):
    """GPT Transformer block."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask)[0])
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
```

---

## 6. Efficient Transformers

### Flash Attention

```python
"""
Flash Attention: IO-aware exact attention algorithm

Standard attention: O(n²) memory (store full attention matrix)
Flash Attention: O(n) memory (tile-based computation)

Key ideas:
1. Block-sparse computation
2. Kernel fusion (reduce memory reads/writes)
3. Recomputation in backward pass

Available via:
- flash_attn library
- PyTorch 2.0+ scaled_dot_product_attention
"""

# Using PyTorch 2.0 Flash Attention
import torch.nn.functional as F

def efficient_attention(query, key, value, mask=None, dropout_p=0.0):
    """PyTorch 2.0+ Flash Attention."""
    return F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=mask,
        dropout_p=dropout_p,
        is_causal=False  # Set True for decoder
    )
```

### Linear Attention Variants

```python
class LinearAttention(nn.Module):
    """
    Linear attention approximation: O(n) instead of O(n²)
    
    Standard: softmax(QK^T)V
    Linear: φ(Q)(φ(K)^T V)
    
    Change order of operations using associativity.
    """
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def feature_map(self, x):
        """ELU + 1 feature map."""
        return F.elu(x) + 1
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        Q = self.feature_map(Q)
        K = self.feature_map(K)
        
        # Linear attention: (QK^T)V = Q(K^T V)
        KV = torch.einsum('bshd,bshm->bhdm', K, V)  # (batch, heads, d_k, d_k)
        Z = 1 / (torch.einsum('bshd,bhd->bsh', Q, K.sum(dim=1)) + 1e-6)
        
        output = torch.einsum('bshd,bhdm,bsh->bshm', Q, KV, Z)
        output = output.reshape(batch_size, seq_len, -1)
        
        return self.W_o(output)
```

### KV Cache for Generation

```python
class CachedMultiHeadAttention(nn.Module):
    """
    Multi-head attention with KV cache for efficient generation.
    
    During generation, we only need to compute attention for new tokens.
    Cache previous K, V to avoid recomputation.
    """
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x, kv_cache=None, use_cache=False):
        batch_size = x.size(0)
        
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        if kv_cache is not None:
            # Concatenate with cached K, V
            past_k, past_v = kv_cache
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)
        
        # Compute attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V)
        
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.d_k * self.num_heads)
        output = self.W_o(attn_output)
        
        if use_cache:
            return output, (K, V)
        return output, None
```

---

## 7. Interview Questions

```python
# Q1: Why do Transformers use Layer Normalization instead of Batch Normalization?
"""
1. Batch independence: LayerNorm normalizes each sample independently
   - No dependency on batch statistics
   - Works with any batch size

2. Sequence length variation: NLP has variable lengths
   - BatchNorm across sequence positions doesn't make sense
   - LayerNorm normalizes across features for each position

3. Autoregressive generation: Can't use future statistics
   - LayerNorm only needs current sample
"""


# Q2: Explain the purpose of the scaling factor √d_k in attention
"""
Without scaling, dot products grow with dimension:
- If Q, K have unit variance, Q·K has variance d_k
- Large values push softmax into saturation
- Saturated softmax → near-zero gradients

With scaling √d_k:
- Q·K / √d_k has unit variance
- Softmax stays in good gradient region
- More stable training
"""


# Q3: What is the difference between Pre-LN and Post-LN Transformers?
"""
Post-LN (Original):
x = x + Attention(x)
x = LayerNorm(x)
- Gradients can be unstable
- Requires careful LR warmup

Pre-LN (Modern):
x = x + Attention(LayerNorm(x))
- More stable training
- Residual path is unmodified
- Used in GPT-2, LLaMA, etc.
"""


# Q4: How does BERT differ from GPT architecturally?
"""
BERT (Encoder-only):
- Bidirectional attention (sees full context)
- Pre-trained with MLM (masked language model)
- Good for: classification, NER, QA
- Cannot generate text autoregressively

GPT (Decoder-only):
- Unidirectional/causal attention (left-to-right)
- Pre-trained with next token prediction
- Good for: text generation, few-shot learning
- Natural for autoregressive generation
"""


# Q5: What are the advantages of RoPE over sinusoidal positional encoding?
"""
RoPE advantages:
1. Relative position in attention: Position encoded in Q·K
2. Better extrapolation: Handles longer sequences than training
3. Applied to Q, K only: More flexible than additive embedding
4. Decays with distance: Natural relative position bias

Sinusoidal limitations:
1. Absolute position: Hard to use relative positions
2. Fixed extrapolation: May not generalize to longer sequences
"""


# Q6: Why is KV caching important for generation?
"""
Without cache:
- Each new token requires full forward pass
- Recomputes all K, V for all previous tokens
- O(n²) attention computation for n tokens

With cache:
- Store K, V from previous steps
- Only compute K, V for new token
- Attention: new Q against all cached K, V
- O(n) per step instead of O(n²)

Memory trade-off: Cache size grows with sequence length
"""


# Q7: Explain the computational complexity of Transformers
"""
Self-attention: O(n² · d)
- n² attention weights, each computed in O(d)

Feed-forward: O(n · d · d_ff)
- Each position: d → d_ff → d

Total per layer: O(n² · d + n · d · d_ff)

Memory:
- Attention matrix: O(n²)
- Activations: O(n · d)

Bottleneck for long sequences: O(n²) attention
Solutions: Linear attention, sparse attention, Flash attention
"""


# Q8: What is the purpose of weight tying in language models?
"""
Weight tying: Share input embedding with output projection
output_projection.weight = input_embedding.weight

Benefits:
1. Fewer parameters (vocab_size × d_model saved)
2. Better generalization
3. Semantic consistency between input/output

Used in: GPT, T5, most modern LLMs
"""


# Q9: How does Flash Attention achieve memory efficiency?
"""
Standard attention stores full n×n attention matrix.

Flash Attention:
1. Tile-based computation: Process blocks of Q, K, V
2. Online softmax: Accumulate numerator and denominator
3. Recomputation: Recompute attention in backward pass
4. Kernel fusion: Minimize GPU memory bandwidth

Result:
- O(n) memory instead of O(n²)
- 2-4x faster due to reduced memory access
- Exact computation (not approximate)
"""


# Q10: Compare encoder-only, decoder-only, and encoder-decoder architectures
"""
Encoder-only (BERT):
- Bidirectional attention
- Pre-training: MLM
- Tasks: Classification, NER, sentiment
- Cannot generate sequentially

Decoder-only (GPT):
- Causal attention
- Pre-training: Next token prediction
- Tasks: Generation, few-shot learning
- Most scalable architecture (GPT-4, LLaMA)

Encoder-Decoder (T5, BART):
- Encoder: Bidirectional
- Decoder: Causal + cross-attention
- Pre-training: Span corruption
- Tasks: Translation, summarization
"""
```
