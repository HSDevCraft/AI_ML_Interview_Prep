# RNNs & Sequence Models - Complete Guide

## ⚡ Interview Quick Summary

> **Core insight**: RNNs process sequences with shared weights but suffer from vanishing gradients over long sequences. LSTM/GRU add gating mechanisms to selectively remember and forget. Transformers replaced them for most tasks via direct attention.

### RNN vs LSTM vs GRU Comparison

| Model | Gates | Handles long deps? | Parameters | Best For |
|-------|-------|-------------------|------------|----------|
| RNN | None | ✗ No (vanishing) | Fewer | Very short sequences |
| LSTM | 3 (forget, input, output) | ✓ Yes | Most | Long sequences, complex patterns |
| GRU | 2 (reset, update) | ✓ Yes (nearly) | Middle | When LSTM is overkill |
| Transformer | Attention (no recurrence) | ✓ Yes (direct) | Most | Parallelizable, NLP |

### LSTM Gating Mechanism — Must Know

```
LSTM has two states: cell state c_t (long-term memory) and hidden state h_t (output)

Forget gate: f_t = σ(W_f[h_{t-1}, x_t] + b_f)    ← what to forget from c_{t-1}
Input gate:  i_t = σ(W_i[h_{t-1}, x_t] + b_i)    ← what new info to add
Candidate:   c̃_t = tanh(W_c[h_{t-1}, x_t] + b_c)  ← new candidate values
Output gate: o_t = σ(W_o[h_{t-1}, x_t] + b_o)    ← what to output

Update:
  c_t = f_t ⨀ c_{t-1} + i_t ⨀ c̃_t   ← ADDITIVE update (not multiplicative!) → no vanishing!
  h_t = o_t ⨀ tanh(c_t)

Why additive update solves vanishing gradient:
  Gradient of c_t w.r.t. c_{t-1} = f_t  (the forget gate)
  When f_t ≈ 1 (learn to keep), gradient flows perfectly through cell state
  The cell state is a gradient highway!
```

### GRU Simplified (2 gates, nearly same performance)

```
Reset gate:  r_t = σ(W_r[h_{t-1}, x_t])        ← how much of past to use
Update gate: z_t = σ(W_z[h_{t-1}, x_t])        ← how much to update (like forget+input)
Candidate:   h̃_t = tanh(W[r_t⨀h_{t-1}, x_t])    ← new candidate with reset gate applied
Update:      h_t = (1-z_t)⨀h_{t-1} + z_t⨀h̃_t   ← interpolate between old and new

GRU vs LSTM:
  GRU:  Simpler, fewer params, often matches LSTM performance
  LSTM: Better for very long sequences, explicit cell state separation
  Practical: Try GRU first, switch to LSTM if performance insufficient
```

### 🚨 Top Interview Pitfalls
- Saying LSTM "solves" vanishing gradient — it **mitigates** it via additive cell update; still struggles for very long sequences (1000+ steps)
- Not knowing that **bidirectional RNNs** use TWO RNNs (forward + backward) and concatenate hidden states — better for understanding, can't use for generation
- Forgetting that **Transformers didn't kill RNNs for everything** — RNNs are still used for streaming (online) inference where full sequence isn't available at once
- Confusing the **hidden state h_t** (output) with **cell state c_t** (internal memory) in LSTM

---

## Table of Contents
1. [Vanilla RNN](#vanilla-rnn)
2. [LSTM](#lstm)
3. [GRU](#gru)
4. [Advanced RNN Architectures](#advanced-rnn-architectures)
5. [Sequence-to-Sequence Models](#sequence-to-sequence-models)
6. [Attention Mechanisms](#attention-mechanisms)
7. [Interview Questions](#interview-questions)

---

## 1. Vanilla RNN

### RNN Fundamentals

```python
import numpy as np

class VanillaRNN:
    """
    Recurrent Neural Network - processes sequences with shared weights.
    
    Hidden state: h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
    Output: y_t = W_hy @ h_t + b_y
    
    Key property: Same weights used at every time step.
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        # Xavier initialization
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        
        self.W_xh = np.random.randn(input_size, hidden_size) * scale
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale
        self.W_hy = np.random.randn(hidden_size, output_size) * scale
        self.b_h = np.zeros((1, hidden_size))
        self.b_y = np.zeros((1, output_size))
        
        self.hidden_size = hidden_size
    
    def forward(self, inputs, h_prev=None):
        """
        Forward pass through sequence.
        
        inputs: List of input vectors [(batch, input_size), ...]
        h_prev: Initial hidden state (batch, hidden_size)
        
        Returns: outputs, final_hidden
        """
        if h_prev is None:
            h_prev = np.zeros((inputs[0].shape[0], self.hidden_size))
        
        self.inputs = inputs
        self.hidden_states = {-1: h_prev}
        self.outputs = []
        
        for t, x_t in enumerate(inputs):
            # Hidden state update
            h_t = np.tanh(
                x_t @ self.W_xh + 
                self.hidden_states[t-1] @ self.W_hh + 
                self.b_h
            )
            self.hidden_states[t] = h_t
            
            # Output
            y_t = h_t @ self.W_hy + self.b_y
            self.outputs.append(y_t)
        
        return self.outputs, self.hidden_states[len(inputs)-1]
    
    def backward(self, d_outputs, clip_value=5.0):
        """
        Backpropagation Through Time (BPTT).
        
        Gradients flow backward through all time steps.
        """
        T = len(d_outputs)
        
        # Initialize gradients
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        # Gradient from future time steps
        dh_next = np.zeros_like(self.hidden_states[0])
        
        for t in reversed(range(T)):
            # Output gradient
            dy = d_outputs[t]
            dW_hy += self.hidden_states[t].T @ dy
            db_y += np.sum(dy, axis=0, keepdims=True)
            
            # Hidden state gradient
            dh = dy @ self.W_hy.T + dh_next
            
            # Gradient through tanh
            dh_raw = dh * (1 - self.hidden_states[t] ** 2)
            
            # Weight gradients
            dW_xh += self.inputs[t].T @ dh_raw
            dW_hh += self.hidden_states[t-1].T @ dh_raw
            db_h += np.sum(dh_raw, axis=0, keepdims=True)
            
            # Gradient to previous hidden state
            dh_next = dh_raw @ self.W_hh.T
        
        # Gradient clipping
        for grad in [dW_xh, dW_hh, dW_hy, db_h, db_y]:
            np.clip(grad, -clip_value, clip_value, out=grad)
        
        return dW_xh, dW_hh, dW_hy, db_h, db_y


# RNN Problems
"""
Vanishing Gradients:
- Gradients shrink exponentially through time
- tanh derivative: max 1, typically < 1
- Product of many small values → 0
- Cannot learn long-term dependencies

Exploding Gradients:
- Gradients grow exponentially
- Large weight updates → unstable training
- Solution: Gradient clipping

Why it happens:
∂h_T/∂h_0 = ∏_{t=1}^T ∂h_t/∂h_{t-1}
           = ∏_{t=1}^T W_hh · diag(tanh'(z_t))

If largest eigenvalue of W_hh > 1: exploding
If largest eigenvalue of W_hh < 1: vanishing
"""
```

### Truncated BPTT

```python
def truncated_bptt(model, sequences, truncation_length=20):
    """
    Truncated Backpropagation Through Time.
    
    Instead of backpropagating through entire sequence:
    - Divide into chunks of truncation_length
    - Backprop within each chunk
    - Detach hidden state between chunks
    
    Reduces memory and computation while maintaining long-term state.
    """
    hidden = None
    total_loss = 0
    
    for chunk_start in range(0, len(sequences), truncation_length):
        chunk = sequences[chunk_start:chunk_start + truncation_length]
        
        # Detach hidden state (stop gradient flow)
        if hidden is not None:
            hidden = hidden.detach()
        
        # Forward pass
        outputs, hidden = model.forward(chunk, hidden)
        
        # Compute loss and backward (only through this chunk)
        loss = compute_loss(outputs, targets[chunk_start:chunk_start + truncation_length])
        loss.backward()
        
        total_loss += loss.item()
    
    return total_loss
```

---

## 2. LSTM

### LSTM Architecture

```python
import numpy as np

class LSTM:
    """
    Long Short-Term Memory - solves vanishing gradient via gating.
    
    Key innovations:
    1. Cell state (c_t): Highway for information flow
    2. Forget gate: What to discard from cell state
    3. Input gate: What new information to store
    4. Output gate: What to output from cell state
    
    Equations:
    f_t = σ(W_f @ [h_{t-1}, x_t] + b_f)    # Forget gate
    i_t = σ(W_i @ [h_{t-1}, x_t] + b_i)    # Input gate
    c̃_t = tanh(W_c @ [h_{t-1}, x_t] + b_c) # Candidate cell state
    c_t = f_t * c_{t-1} + i_t * c̃_t        # Cell state update
    o_t = σ(W_o @ [h_{t-1}, x_t] + b_o)    # Output gate
    h_t = o_t * tanh(c_t)                   # Hidden state
    """
    
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        
        # Combined weight matrix for efficiency: [f, i, c̃, o]
        combined_size = input_size + hidden_size
        self.W = np.random.randn(combined_size, 4 * hidden_size) * 0.01
        self.b = np.zeros((1, 4 * hidden_size))
        
        # Initialize forget gate bias to 1 (remember by default)
        self.b[0, :hidden_size] = 1.0
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward_step(self, x_t, h_prev, c_prev):
        """Single LSTM step."""
        # Concatenate input and previous hidden
        combined = np.concatenate([h_prev, x_t], axis=1)
        
        # Compute all gates at once
        gates = combined @ self.W + self.b
        
        # Split into individual gates
        hs = self.hidden_size
        f_t = self.sigmoid(gates[:, :hs])           # Forget
        i_t = self.sigmoid(gates[:, hs:2*hs])       # Input
        c_tilde = np.tanh(gates[:, 2*hs:3*hs])      # Candidate
        o_t = self.sigmoid(gates[:, 3*hs:])         # Output
        
        # Cell state update (additive, not multiplicative!)
        c_t = f_t * c_prev + i_t * c_tilde
        
        # Hidden state
        h_t = o_t * np.tanh(c_t)
        
        # Cache for backward pass
        self.cache = (x_t, h_prev, c_prev, f_t, i_t, c_tilde, o_t, c_t, h_t)
        
        return h_t, c_t
    
    def forward(self, inputs, h0=None, c0=None):
        """Forward through entire sequence."""
        batch_size = inputs[0].shape[0]
        
        if h0 is None:
            h0 = np.zeros((batch_size, self.hidden_size))
        if c0 is None:
            c0 = np.zeros((batch_size, self.hidden_size))
        
        h, c = h0, c0
        hidden_states = []
        
        for x_t in inputs:
            h, c = self.forward_step(x_t, h, c)
            hidden_states.append(h)
        
        return hidden_states, (h, c)


# Why LSTM solves vanishing gradient
"""
Cell state gradient:
∂c_T/∂c_0 = ∏_{t=1}^T f_t

If f_t ≈ 1 (remember everything):
- Gradient flows unchanged through time
- No vanishing!

Additive update (c_t = f_t*c_{t-1} + ...) vs multiplicative (h_t = W*h_{t-1})
- Additive preserves gradient magnitude
- Forget gate controls gradient flow

Key insight: Identity mapping as default behavior
"""
```

### Peephole LSTM

```python
class PeepholeLSTM:
    """
    LSTM with peephole connections.
    
    Gates look at cell state directly:
    f_t = σ(W_f @ [h_{t-1}, x_t] + W_cf * c_{t-1} + b_f)
    i_t = σ(W_i @ [h_{t-1}, x_t] + W_ci * c_{t-1} + b_i)
    o_t = σ(W_o @ [h_{t-1}, x_t] + W_co * c_t + b_o)
    
    Allows gates to use timing information from cell state.
    """
    
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        
        # Standard weights
        combined = input_size + hidden_size
        self.W = np.random.randn(combined, 4 * hidden_size) * 0.01
        self.b = np.zeros((1, 4 * hidden_size))
        
        # Peephole weights (diagonal connections)
        self.W_cf = np.random.randn(hidden_size) * 0.01  # Forget peephole
        self.W_ci = np.random.randn(hidden_size) * 0.01  # Input peephole
        self.W_co = np.random.randn(hidden_size) * 0.01  # Output peephole
```

---

## 3. GRU

### GRU Architecture

```python
import numpy as np

class GRU:
    """
    Gated Recurrent Unit - simplified LSTM.
    
    Key differences from LSTM:
    - No separate cell state (only hidden state)
    - Two gates instead of three (reset, update)
    - Fewer parameters, similar performance
    
    Equations:
    z_t = σ(W_z @ [h_{t-1}, x_t] + b_z)    # Update gate
    r_t = σ(W_r @ [h_{t-1}, x_t] + b_r)    # Reset gate
    h̃_t = tanh(W_h @ [r_t * h_{t-1}, x_t] + b_h)  # Candidate
    h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t  # Hidden state
    """
    
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        
        # Weight matrices for update and reset gates
        combined = input_size + hidden_size
        self.W_z = np.random.randn(combined, hidden_size) * 0.01
        self.W_r = np.random.randn(combined, hidden_size) * 0.01
        self.W_h = np.random.randn(combined, hidden_size) * 0.01
        
        self.b_z = np.zeros((1, hidden_size))
        self.b_r = np.zeros((1, hidden_size))
        self.b_h = np.zeros((1, hidden_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward_step(self, x_t, h_prev):
        """Single GRU step."""
        # Concatenate input and previous hidden
        combined = np.concatenate([h_prev, x_t], axis=1)
        
        # Update gate: controls how much of past to keep
        z_t = self.sigmoid(combined @ self.W_z + self.b_z)
        
        # Reset gate: controls how much of past to forget for candidate
        r_t = self.sigmoid(combined @ self.W_r + self.b_r)
        
        # Candidate hidden state
        combined_reset = np.concatenate([r_t * h_prev, x_t], axis=1)
        h_tilde = np.tanh(combined_reset @ self.W_h + self.b_h)
        
        # Final hidden state: interpolate between old and new
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        
        return h_t
    
    def forward(self, inputs, h0=None):
        """Forward through entire sequence."""
        batch_size = inputs[0].shape[0]
        
        if h0 is None:
            h0 = np.zeros((batch_size, self.hidden_size))
        
        h = h0
        hidden_states = []
        
        for x_t in inputs:
            h = self.forward_step(x_t, h)
            hidden_states.append(h)
        
        return hidden_states, h


# GRU vs LSTM comparison
"""
GRU:
- 2 gates (update, reset)
- No cell state
- Fewer parameters (~75% of LSTM)
- Faster training
- Works well on smaller datasets

LSTM:
- 3 gates (forget, input, output)
- Separate cell state
- More parameters
- Better for very long sequences
- More expressive

When to use:
- Default: LSTM (more well-studied)
- Smaller data / faster training: GRU
- Very long sequences: LSTM
- Both work well in practice!
"""
```

---

## 4. Advanced RNN Architectures

### Bidirectional RNN

```python
import torch
import torch.nn as nn

class BidirectionalRNN(nn.Module):
    """
    Process sequence in both directions.
    
    Captures both past and future context.
    Output: concatenation of forward and backward hidden states.
    
    Use cases:
    - Sequence labeling (NER, POS tagging)
    - Sentiment analysis
    - Machine translation (encoder)
    """
    
    def __init__(self, input_size, hidden_size, num_layers=1, rnn_type='lstm'):
        super().__init__()
        
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                              batch_first=True, bidirectional=True)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                             batch_first=True, bidirectional=True)
        
        self.hidden_size = hidden_size
    
    def forward(self, x, lengths=None):
        """
        x: (batch, seq_len, input_size)
        lengths: actual sequence lengths (for packed sequences)
        """
        if lengths is not None:
            # Pack padded sequence for efficiency
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        output, hidden = self.rnn(x)
        
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        # output shape: (batch, seq_len, 2 * hidden_size)
        # hidden shape: (2 * num_layers, batch, hidden_size)
        
        return output, hidden


# Manual bidirectional implementation
class ManualBidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.forward_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.backward_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        # Forward pass
        out_forward, _ = self.forward_lstm(x)
        
        # Backward pass (reverse input)
        x_reversed = torch.flip(x, [1])
        out_backward, _ = self.backward_lstm(x_reversed)
        out_backward = torch.flip(out_backward, [1])
        
        # Concatenate
        return torch.cat([out_forward, out_backward], dim=-1)
```

### Stacked/Deep RNN

```python
class StackedLSTM(nn.Module):
    """
    Multiple LSTM layers stacked vertically.
    
    Each layer takes previous layer's output as input.
    Deeper networks can learn more complex patterns.
    
    Common: 2-4 layers for NLP tasks
    """
    
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
    
    def forward(self, x, hidden=None):
        return self.lstm(x, hidden)


# Residual connections for deep RNNs
class ResidualLSTM(nn.Module):
    """Add skip connections between LSTM layers."""
    
    def __init__(self, input_size, hidden_size, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_size = input_size if i == 0 else hidden_size
            self.layers.append(nn.LSTM(in_size, hidden_size, batch_first=True))
        
        # Projection for first layer (if input_size != hidden_size)
        self.input_proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
    
    def forward(self, x):
        if self.input_proj:
            residual = self.input_proj(x)
        else:
            residual = x
        
        for i, layer in enumerate(self.layers):
            out, _ = layer(x if i == 0 else out)
            if i > 0:  # Skip connection
                out = out + residual
            residual = out
        
        return out
```

### Variable Length Sequences

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def process_variable_length(sequences, lengths, rnn):
    """
    Efficiently process sequences of different lengths.
    
    1. Pad sequences to same length
    2. Pack (remove padding for RNN)
    3. Process through RNN
    4. Unpack (add padding back)
    """
    # Sort by length (required for pack_padded_sequence)
    sorted_lengths, sorted_idx = lengths.sort(descending=True)
    sorted_sequences = sequences[sorted_idx]
    
    # Pack
    packed = pack_padded_sequence(
        sorted_sequences, 
        sorted_lengths.cpu(), 
        batch_first=True
    )
    
    # RNN forward
    packed_output, hidden = rnn(packed)
    
    # Unpack
    output, _ = pad_packed_sequence(packed_output, batch_first=True)
    
    # Unsort
    _, unsorted_idx = sorted_idx.sort()
    output = output[unsorted_idx]
    
    return output, hidden


# Getting last valid output for each sequence
def get_last_output(output, lengths):
    """
    Extract the last non-padded output for each sequence.
    
    output: (batch, max_len, hidden_size)
    lengths: (batch,) - actual lengths
    """
    batch_size = output.size(0)
    # Gather last valid outputs
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, output.size(2))
    last_output = output.gather(1, idx).squeeze(1)
    return last_output
```

---

## 5. Sequence-to-Sequence Models

### Encoder-Decoder Architecture

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encodes input sequence into fixed-size context vector.
    """
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
    
    def forward(self, x, lengths=None):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_size)
        
        if lengths is not None:
            embedded = pack_padded_sequence(embedded, lengths.cpu(), 
                                           batch_first=True, enforce_sorted=False)
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        if lengths is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        
        return outputs, (hidden, cell)


class Decoder(nn.Module):
    """
    Generates output sequence from context vector.
    """
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden):
        # x: (batch, 1) - single token
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc(output)
        return prediction, hidden


class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence model for tasks like translation.
    """
    
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: (batch, src_len)
        trg: (batch, trg_len)
        """
        batch_size = trg.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.fc.out_features
        
        # Store outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encode
        _, hidden = self.encoder(src)
        
        # First decoder input is <SOS>
        decoder_input = trg[:, 0:1]
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(decoder_input, hidden)
            outputs[:, t:t+1, :] = output
            
            # Teacher forcing
            if torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = trg[:, t:t+1]
            else:
                decoder_input = output.argmax(dim=-1)
        
        return outputs
    
    def generate(self, src, max_len=50, sos_token=1, eos_token=2):
        """Generate sequence (inference mode)."""
        self.eval()
        with torch.no_grad():
            _, hidden = self.encoder(src)
            
            decoder_input = torch.tensor([[sos_token]]).to(self.device)
            generated = [sos_token]
            
            for _ in range(max_len):
                output, hidden = self.decoder(decoder_input, hidden)
                predicted = output.argmax(dim=-1)
                generated.append(predicted.item())
                
                if predicted.item() == eos_token:
                    break
                
                decoder_input = predicted
        
        return generated
```

### Beam Search

```python
def beam_search(model, src, beam_width=5, max_len=50, sos_token=1, eos_token=2):
    """
    Beam search decoding - keeps top-k hypotheses at each step.
    
    Better than greedy search for sequence generation.
    """
    device = src.device
    
    # Encode
    _, hidden = model.encoder(src)
    
    # Initialize beams: (score, sequence, hidden_state)
    beams = [(0.0, [sos_token], hidden)]
    
    for _ in range(max_len):
        candidates = []
        
        for score, seq, hidden in beams:
            if seq[-1] == eos_token:
                candidates.append((score, seq, hidden))
                continue
            
            # Decode one step
            decoder_input = torch.tensor([[seq[-1]]]).to(device)
            output, new_hidden = model.decoder(decoder_input, hidden)
            log_probs = torch.log_softmax(output.squeeze(), dim=-1)
            
            # Get top-k tokens
            top_probs, top_indices = log_probs.topk(beam_width)
            
            for prob, idx in zip(top_probs, top_indices):
                new_score = score + prob.item()
                new_seq = seq + [idx.item()]
                candidates.append((new_score, new_seq, new_hidden))
        
        # Keep top beam_width candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]
        
        # Early stopping if all beams ended
        if all(beam[1][-1] == eos_token for beam in beams):
            break
    
    # Return best sequence
    best_beam = max(beams, key=lambda x: x[0] / len(x[1]))  # Length normalization
    return best_beam[1]
```

---

## 6. Attention Mechanisms

### Bahdanau Attention (Additive)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """
    Additive attention (Bahdanau et al., 2015).
    
    score(h_t, h_s) = v^T tanh(W_q h_t + W_k h_s)
    
    Computes alignment between decoder state and all encoder states.
    """
    
    def __init__(self, hidden_size):
        super().__init__()
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, query, keys, values, mask=None):
        """
        query: (batch, hidden) - decoder hidden state
        keys: (batch, seq_len, hidden) - encoder outputs
        values: (batch, seq_len, hidden) - encoder outputs (same as keys)
        mask: (batch, seq_len) - 0 for padded positions
        """
        # Expand query for broadcasting
        query = query.unsqueeze(1)  # (batch, 1, hidden)
        
        # Compute attention scores
        scores = self.v(torch.tanh(self.W_q(query) + self.W_k(keys)))
        scores = scores.squeeze(-1)  # (batch, seq_len)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
        
        # Weighted sum of values
        context = torch.bmm(attn_weights.unsqueeze(1), values)
        context = context.squeeze(1)  # (batch, hidden)
        
        return context, attn_weights


class AttentionDecoder(nn.Module):
    """Decoder with attention."""
    
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = BahdanauAttention(hidden_size)
        self.rnn = nn.LSTM(embed_size + hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
    
    def forward(self, x, hidden, encoder_outputs, mask=None):
        # x: (batch, 1)
        embedded = self.embedding(x)  # (batch, 1, embed_size)
        
        # Get context from attention
        h = hidden[0][-1]  # Last layer hidden state
        context, attn_weights = self.attention(h, encoder_outputs, encoder_outputs, mask)
        
        # Concatenate embedding and context
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        
        # RNN step
        output, hidden = self.rnn(rnn_input, hidden)
        
        # Predict
        output = torch.cat([output.squeeze(1), context], dim=-1)
        prediction = self.fc(output)
        
        return prediction, hidden, attn_weights
```

### Luong Attention (Multiplicative)

```python
class LuongAttention(nn.Module):
    """
    Multiplicative attention (Luong et al., 2015).
    
    Three variants:
    - dot: score = h_t^T h_s
    - general: score = h_t^T W h_s
    - concat: score = v^T tanh(W[h_t; h_s])
    """
    
    def __init__(self, hidden_size, method='general'):
        super().__init__()
        self.method = method
        
        if method == 'general':
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == 'concat':
            self.W = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, query, keys, mask=None):
        """
        query: (batch, hidden)
        keys: (batch, seq_len, hidden)
        """
        if self.method == 'dot':
            scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)
        
        elif self.method == 'general':
            scores = torch.bmm(keys, self.W(query).unsqueeze(-1)).squeeze(-1)
        
        elif self.method == 'concat':
            query_expanded = query.unsqueeze(1).expand(-1, keys.size(1), -1)
            concat = torch.cat([query_expanded, keys], dim=-1)
            scores = self.v(torch.tanh(self.W(concat))).squeeze(-1)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), keys).squeeze(1)
        
        return context, attn_weights
```

### Multi-Head Attention (Preview)

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (Vaswani et al., 2017).
    
    Allows model to attend to different representation subspaces.
    Foundation of Transformer architecture.
    """
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, V), attn_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.W_o(attn_output)
```

---

## 7. Interview Questions

```python
# Q1: Why does vanilla RNN suffer from vanishing gradient?
"""
Gradient through time:
∂h_T/∂h_0 = ∏_{t=1}^T W_hh · diag(tanh'(z_t))

Problems:
1. tanh derivative is at most 1, usually < 1
2. If ||W_hh|| < 1, gradients shrink exponentially
3. Long sequences → many multiplications → gradient → 0

Result: Cannot learn long-term dependencies
"""


# Q2: How does LSTM solve the vanishing gradient problem?
"""
Key innovations:
1. Cell state: Highway for information (additive updates)
   c_t = f_t * c_{t-1} + i_t * c̃_t
   
2. Gradient through cell state:
   ∂c_T/∂c_0 = ∏_{t=1}^T f_t
   If f_t ≈ 1, gradient ≈ 1 (no vanishing!)

3. Gates control gradient flow:
   - Forget gate near 1: preserve gradient
   - Forget gate near 0: reset memory

4. Additive vs multiplicative:
   RNN: h_t = f(W·h_{t-1}) → multiplicative
   LSTM: c_t = f_t·c_{t-1} + ... → additive preserves gradient
"""


# Q3: What is teacher forcing and its pros/cons?
"""
Teacher forcing: During training, use ground truth as next input
instead of model's prediction.

Pros:
- Faster convergence
- More stable training
- Model sees correct context

Cons:
- Exposure bias: Training/inference mismatch
- Model may not recover from errors at inference

Solutions:
- Scheduled sampling: Gradually reduce teacher forcing
- Curriculum learning: Start with teacher forcing, decrease
"""


# Q4: LSTM vs GRU - when to use which?
"""
GRU:
- Fewer parameters (2 gates vs 3)
- Faster training
- Works well on smaller datasets
- Similar performance to LSTM

LSTM:
- More parameters, more expressive
- Separate cell state
- Better for very long sequences
- More studied, better understood

Rule of thumb:
- Start with GRU (faster experimentation)
- Use LSTM if GRU doesn't work well
- Both: try both and compare
"""


# Q5: What is the difference between Bahdanau and Luong attention?
"""
Bahdanau (Additive):
- score = v^T tanh(W_q·query + W_k·key)
- Computed before RNN step (input to decoder)
- More parameters

Luong (Multiplicative):
- score = query^T · W · key (or just dot product)
- Computed after RNN step (concat with output)
- Fewer parameters, faster

Performance: Similar, Luong often preferred for simplicity
"""


# Q6: Why use attention instead of just the final encoder state?
"""
Problems with final state only:
1. Information bottleneck: All info compressed into one vector
2. Long sequences: Hard to capture everything
3. No alignment: Decoder can't focus on relevant parts

Attention benefits:
1. Dynamic context: Different context for each output
2. No bottleneck: Access all encoder states
3. Alignment: Model learns where to look
4. Interpretability: Can visualize attention weights
"""


# Q7: What is bidirectional RNN and when to use it?
"""
Bidirectional RNN:
- Two RNNs: forward (left-to-right) and backward (right-to-left)
- Output: concatenation of both hidden states
- Captures both past and future context

Use when:
- Full sequence available (not online/streaming)
- Task benefits from future context

Examples:
- Sequence labeling (NER, POS)
- Machine translation encoder
- Sentiment analysis

Don't use for:
- Language modeling (predict next token)
- Real-time/streaming applications
"""


# Q8: How to handle variable-length sequences in RNNs?
"""
1. Padding + masking:
   - Pad to max length
   - Create mask tensor
   - Apply mask to loss calculation

2. Packed sequences (PyTorch):
   - pack_padded_sequence: Remove padding before RNN
   - pad_packed_sequence: Add padding back after
   - More efficient computation

3. Bucketing:
   - Group similar-length sequences in batches
   - Reduces padding overhead

4. Dynamic batching:
   - Sort by length
   - Variable batch sizes
"""


# Q9: What is the bottleneck problem in seq2seq?
"""
Problem:
- Encoder compresses entire input to fixed-size vector
- All information must fit in this "bottleneck"
- Longer sequences → more compression → information loss

Solutions:
1. Attention: Access all encoder states, not just final
2. Larger hidden size (limited help)
3. Bidirectional encoder
4. Transformer (no recurrence, parallel attention)
"""


# Q10: How does beam search improve over greedy decoding?
"""
Greedy decoding:
- Choose highest probability token at each step
- May miss globally optimal sequence
- Fast but suboptimal

Beam search:
- Keep top-k hypotheses at each step
- Explore multiple paths
- Better global solutions
- Trade-off: slower (k times more computation)

Improvements:
- Length normalization: Prevent preference for short sequences
- Coverage penalty: Encourage attending to all input
- Early stopping: Stop when top hypothesis ends
"""
```
