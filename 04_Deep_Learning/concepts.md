# Deep Learning - Complete Guide

## 1. Neural Network Fundamentals

### Perceptron & MLP

```python
import numpy as np

class Perceptron:
    """Single-layer perceptron"""
    def __init__(self, n_features, lr=0.01):
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.lr = lr
    
    def activation(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, X):
        linear = X @ self.weights + self.bias
        return np.array([self.activation(x) for x in linear])
    
    def fit(self, X, y, epochs=100):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                pred = self.activation(xi @ self.weights + self.bias)
                update = self.lr * (yi - pred)
                self.weights += update * xi
                self.bias += update


class MLP:
    """Multi-Layer Perceptron from scratch"""
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.layers.append({'w': w, 'b': b})
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i, layer in enumerate(self.layers):
            z = self.activations[-1] @ layer['w'] + layer['b']
            self.z_values.append(z)
            
            if i == len(self.layers) - 1:
                a = self.softmax(z)
            else:
                a = self.relu(z)
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, y, lr=0.01):
        m = y.shape[0]
        n_layers = len(self.layers)
        
        # Output layer gradient (cross-entropy + softmax)
        dz = self.activations[-1] - y
        
        for i in reversed(range(n_layers)):
            dw = (1/m) * self.activations[i].T @ dz
            db = (1/m) * np.sum(dz, axis=0, keepdims=True)
            
            if i > 0:
                da = dz @ self.layers[i]['w'].T
                dz = da * self.relu_derivative(self.z_values[i-1])
            
            # Update weights
            self.layers[i]['w'] -= lr * dw
            self.layers[i]['b'] -= lr * db
```

### Activation Functions

```python
def sigmoid(x):
    """σ(x) = 1 / (1 + e^(-x))
    Range: (0, 1)
    Issues: Vanishing gradient, not zero-centered
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    """tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    Range: (-1, 1)
    Zero-centered, but still vanishing gradient
    """
    return np.tanh(x)

def relu(x):
    """ReLU(x) = max(0, x)
    Range: [0, ∞)
    Issues: Dying ReLU (neurons can "die")
    """
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    """LeakyReLU(x) = max(αx, x)
    Fixes dying ReLU problem
    """
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    """ELU(x) = x if x > 0 else α(e^x - 1)
    Smooth, handles negative values
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def gelu(x):
    """GELU(x) = x * Φ(x)
    Used in BERT, GPT
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def swish(x, beta=1.0):
    """Swish(x) = x * sigmoid(βx)
    Self-gated activation
    """
    return x * sigmoid(beta * x)

def softmax(x):
    """Softmax for multi-class classification
    Converts logits to probabilities
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

### Loss Functions

```python
def mse_loss(y_true, y_pred):
    """Mean Squared Error - Regression"""
    return np.mean((y_true - y_pred) ** 2)

def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """Binary Cross-Entropy - Binary Classification
    L = -[y*log(p) + (1-y)*log(1-p)]
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """Categorical Cross-Entropy - Multi-class Classification
    L = -Σ y_i * log(p_i)
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Focal Loss - Imbalanced Classification
    FL = -α(1-p)^γ * log(p)
    """
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    return -np.mean(alpha * (1 - pt) ** gamma * np.log(pt))

def huber_loss(y_true, y_pred, delta=1.0):
    """Huber Loss - Robust Regression
    Combines MSE and MAE
    """
    error = y_true - y_pred
    is_small = np.abs(error) <= delta
    small_loss = 0.5 * error ** 2
    large_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.mean(np.where(is_small, small_loss, large_loss))
```

### Optimizers

```python
class SGD:
    """Stochastic Gradient Descent with Momentum"""
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None
    
    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        
        for i, (p, g) in enumerate(zip(params, grads)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * g
            p += self.velocity[i]


class Adam:
    """Adam Optimizer
    Combines momentum and RMSprop
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0
    
    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        
        for i, (p, g) in enumerate(zip(params, grads)):
            # Update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class AdamW:
    """AdamW - Adam with decoupled weight decay"""
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        
        for i, (p, g) in enumerate(zip(params, grads)):
            # Weight decay (decoupled)
            p -= self.lr * self.weight_decay * p
            
            # Adam update
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
```

### Weight Initialization

```python
def xavier_uniform(fan_in, fan_out):
    """Xavier/Glorot Uniform - Good for sigmoid/tanh"""
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

def xavier_normal(fan_in, fan_out):
    """Xavier/Glorot Normal"""
    std = np.sqrt(2 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out) * std

def he_uniform(fan_in, fan_out):
    """He/Kaiming Uniform - Good for ReLU"""
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

def he_normal(fan_in, fan_out):
    """He/Kaiming Normal - Good for ReLU"""
    std = np.sqrt(2 / fan_in)
    return np.random.randn(fan_in, fan_out) * std
```

### Regularization Techniques

```python
# L1 Regularization (Lasso)
# Loss = Original_Loss + λ * Σ|w|
# Promotes sparsity

# L2 Regularization (Ridge/Weight Decay)
# Loss = Original_Loss + λ * Σw²
# Prevents large weights

# Dropout
class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
    
    def forward(self, x, training=True):
        if training:
            self.mask = np.random.binomial(1, 1-self.p, x.shape) / (1-self.p)
            return x * self.mask
        return x
    
    def backward(self, grad):
        return grad * self.mask

# Batch Normalization
class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.eps = eps
        self.momentum = momentum
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x, training=True):
        if training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

# Layer Normalization (used in Transformers)
class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

---

## 2. Convolutional Neural Networks (CNNs)

### Convolution Operation

```python
def conv2d(input, kernel, stride=1, padding=0):
    """2D Convolution"""
    # Add padding
    if padding > 0:
        input = np.pad(input, ((padding, padding), (padding, padding)), mode='constant')
    
    h_in, w_in = input.shape
    k_h, k_w = kernel.shape
    
    h_out = (h_in - k_h) // stride + 1
    w_out = (w_in - k_w) // stride + 1
    
    output = np.zeros((h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            region = input[i*stride:i*stride+k_h, j*stride:j*stride+k_w]
            output[i, j] = np.sum(region * kernel)
    
    return output

# Output size formula:
# H_out = (H_in + 2*padding - kernel_size) / stride + 1
```

### Pooling

```python
def max_pool2d(input, pool_size=2, stride=2):
    """Max Pooling"""
    h_in, w_in = input.shape
    h_out = (h_in - pool_size) // stride + 1
    w_out = (w_in - pool_size) // stride + 1
    
    output = np.zeros((h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            region = input[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            output[i, j] = np.max(region)
    
    return output

def avg_pool2d(input, pool_size=2, stride=2):
    """Average Pooling"""
    h_in, w_in = input.shape
    h_out = (h_in - pool_size) // stride + 1
    w_out = (w_in - pool_size) // stride + 1
    
    output = np.zeros((h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            region = input[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            output[i, j] = np.mean(region)
    
    return output

def global_avg_pool(input):
    """Global Average Pooling"""
    return np.mean(input, axis=(1, 2))
```

### CNN Architectures

```python
import torch
import torch.nn as nn

# LeNet-5 (1998)
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# VGG Block
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels, 3, padding=1
            ))
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

# ResNet Block (Residual Connection)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)  # Residual connection
        return torch.relu(out)

# Inception Module (GoogLeNet)
class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, 
                 ch5x5_reduce, ch5x5, pool_proj):
        super().__init__()
        # 1x1 branch
        self.branch1 = nn.Conv2d(in_channels, ch1x1, 1)
        
        # 1x1 -> 3x3 branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, 1),
            nn.Conv2d(ch3x3_reduce, ch3x3, 3, padding=1)
        )
        
        # 1x1 -> 5x5 branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, 1),
            nn.Conv2d(ch5x5_reduce, ch5x5, 5, padding=2)
        )
        
        # MaxPool -> 1x1 branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, 1)
        )
    
    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)
```

### Object Detection

```python
# Key Concepts:

# 1. Region-based methods (R-CNN family)
# - R-CNN: Region proposals -> CNN features -> SVM classifier
# - Fast R-CNN: Single CNN, ROI pooling
# - Faster R-CNN: Region Proposal Network (RPN)

# 2. Single-shot methods
# - YOLO: Single forward pass, grid-based detection
# - SSD: Multi-scale feature maps

# IoU (Intersection over Union)
def iou(box1, box2):
    """
    box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

# Non-Maximum Suppression (NMS)
def nms(boxes, scores, threshold=0.5):
    """Remove overlapping boxes"""
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        ious = np.array([iou(boxes[current], boxes[i]) for i in indices[1:]])
        indices = indices[1:][ious < threshold]
    
    return keep
```

---

## 3. Recurrent Neural Networks (RNNs)

### Vanilla RNN

```python
class RNN:
    """Simple RNN cell"""
    def __init__(self, input_size, hidden_size, output_size):
        # Weight matrices
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(hidden_size, output_size) * 0.01
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))
        self.hidden_size = hidden_size
    
    def forward(self, inputs, h_prev):
        """
        inputs: sequence of input vectors
        h_prev: previous hidden state
        """
        self.inputs = inputs
        self.hs = {-1: h_prev}
        self.ys = {}
        
        for t, x in enumerate(inputs):
            # Hidden state: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
            self.hs[t] = np.tanh(x @ self.Wxh + self.hs[t-1] @ self.Whh + self.bh)
            # Output: y_t = W_hy * h_t + b_y
            self.ys[t] = self.hs[t] @ self.Why + self.by
        
        return self.ys, self.hs[len(inputs)-1]

# Problem: Vanishing/Exploding gradients
# - Gradients shrink/grow exponentially through time
# - Difficult to learn long-term dependencies
```

### LSTM (Long Short-Term Memory)

```python
class LSTM:
    """LSTM cell with gates"""
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        
        # Combined weights for efficiency
        # [forget, input, cell, output] gates
        self.W = np.random.randn(input_size + hidden_size, 4 * hidden_size) * 0.01
        self.b = np.zeros((1, 4 * hidden_size))
    
    def forward(self, x, h_prev, c_prev):
        """
        x: input at time t
        h_prev: previous hidden state
        c_prev: previous cell state
        """
        # Concatenate input and hidden state
        combined = np.concatenate([x, h_prev], axis=1)
        
        # Compute all gates at once
        gates = combined @ self.W + self.b
        
        # Split into individual gates
        f = sigmoid(gates[:, :self.hidden_size])           # Forget gate
        i = sigmoid(gates[:, self.hidden_size:2*self.hidden_size])  # Input gate
        c_tilde = np.tanh(gates[:, 2*self.hidden_size:3*self.hidden_size])  # Candidate
        o = sigmoid(gates[:, 3*self.hidden_size:])         # Output gate
        
        # Cell state update
        c = f * c_prev + i * c_tilde
        
        # Hidden state
        h = o * np.tanh(c)
        
        return h, c

# LSTM solves vanishing gradient through:
# 1. Cell state (highway for gradients)
# 2. Forget gate (controls information flow)
# 3. Additive updates (not multiplicative)
```

### GRU (Gated Recurrent Unit)

```python
class GRU:
    """GRU cell - simpler than LSTM"""
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        
        # Reset and update gate weights
        self.W_z = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.W_r = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
        self.W_h = np.random.randn(input_size + hidden_size, hidden_size) * 0.01
    
    def forward(self, x, h_prev):
        combined = np.concatenate([x, h_prev], axis=1)
        
        # Update gate
        z = sigmoid(combined @ self.W_z)
        
        # Reset gate
        r = sigmoid(combined @ self.W_r)
        
        # Candidate hidden state
        combined_reset = np.concatenate([x, r * h_prev], axis=1)
        h_tilde = np.tanh(combined_reset @ self.W_h)
        
        # Final hidden state
        h = (1 - z) * h_prev + z * h_tilde
        
        return h

# GRU vs LSTM:
# - Fewer parameters (2 gates vs 3)
# - No separate cell state
# - Similar performance in many tasks
```

### Bidirectional RNN

```python
class BidirectionalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.forward_rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.backward_rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    
    def forward(self, x):
        # Forward pass
        out_forward, _ = self.forward_rnn(x)
        
        # Backward pass (reverse input)
        x_reversed = torch.flip(x, [1])
        out_backward, _ = self.backward_rnn(x_reversed)
        out_backward = torch.flip(out_backward, [1])
        
        # Concatenate
        return torch.cat([out_forward, out_backward], dim=-1)

# Use cases: NER, POS tagging, sentiment analysis
```

### Seq2Seq with Attention

```python
class Attention(nn.Module):
    """Bahdanau (Additive) Attention"""
    def __init__(self, hidden_size):
        super().__init__()
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
    
    def forward(self, query, keys, values, mask=None):
        # query: [batch, hidden]
        # keys, values: [batch, seq_len, hidden]
        
        query = self.W_q(query).unsqueeze(1)  # [batch, 1, hidden]
        keys = self.W_k(keys)                  # [batch, seq_len, hidden]
        
        # Attention scores
        scores = self.v(torch.tanh(query + keys))  # [batch, seq_len, 1]
        scores = scores.squeeze(-1)                 # [batch, seq_len]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        weights = torch.softmax(scores, dim=-1)
        
        # Context vector
        context = torch.bmm(weights.unsqueeze(1), values)
        return context.squeeze(1), weights
```

---

## 4. Transformers

### Self-Attention Mechanism

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    
    query, key, value: [batch, seq_len, d_model]
    """
    d_k = query.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask (for decoder self-attention)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention allows attending to different representation subspaces
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
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into heads: [batch, seq_len, d_model] -> [batch, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final projection
        return self.W_o(attn_output)
```

### Positional Encoding

```python
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class RotaryPositionalEmbedding(nn.Module):
    """RoPE - Used in modern LLMs like LLaMA"""
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos', emb.cos())
        self.register_buffer('sin', emb.sin())
    
    def forward(self, x, seq_len):
        return (
            self.cos[:seq_len],
            self.sin[:seq_len]
        )
```

### Transformer Block

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention
        attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
```

### Full Transformer

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 num_heads=8, num_layers=6, d_ff=2048, dropout=0.1, max_len=5000):
        super().__init__()
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model
    
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        
        seq_len = tgt.size(1)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).bool()
        tgt_mask = tgt_mask & causal_mask
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # Encoder
        src_emb = self.positional_encoding(
            self.encoder_embedding(src) * math.sqrt(self.d_model)
        )
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        
        # Decoder
        tgt_emb = self.positional_encoding(
            self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        )
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        
        return self.fc_out(dec_output)
```

---

## 5. Advanced Deep Learning

### Generative Adversarial Networks (GANs)

```python
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)


# GAN Training Loop
def train_gan(generator, discriminator, dataloader, epochs, latent_dim):
    criterion = nn.BCELoss()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    
    for epoch in range(epochs):
        for real_imgs, _ in dataloader:
            batch_size = real_imgs.size(0)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            # Real images
            real_loss = criterion(discriminator(real_imgs), real_labels)
            
            # Fake images
            z = torch.randn(batch_size, latent_dim)
            fake_imgs = generator(z)
            fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels)
            
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            g_loss = criterion(discriminator(fake_imgs), real_labels)
            g_loss.backward()
            optimizer_G.step()
```

### Variational Autoencoder (VAE)

```python
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


def vae_loss(recon_x, x, mu, log_var):
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kl_loss
```

### Transfer Learning & Fine-tuning

```python
import torchvision.models as models

# Load pretrained model
resnet = models.resnet50(pretrained=True)

# Freeze base layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace classifier for new task
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, num_classes)

# Fine-tuning: unfreeze some layers
for param in resnet.layer4.parameters():
    param.requires_grad = True

# Different learning rates
optimizer = torch.optim.Adam([
    {'params': resnet.layer4.parameters(), 'lr': 1e-4},
    {'params': resnet.fc.parameters(), 'lr': 1e-3}
])
```

### Model Compression

```python
# 1. Pruning - Remove unimportant weights
import torch.nn.utils.prune as prune

# Magnitude pruning
prune.l1_unstructured(model.fc1, name='weight', amount=0.3)

# 2. Quantization - Reduce precision
model_quantized = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# 3. Knowledge Distillation
def distillation_loss(student_logits, teacher_logits, labels, T=4, alpha=0.7):
    """
    T: Temperature (higher = softer probabilities)
    alpha: Weight for distillation loss
    """
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    
    hard_loss = F.cross_entropy(student_logits, labels)
    
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

---

## 6. PyTorch Best Practices

```python
# Device handling
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Learning rate scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# or
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, total_steps=len(dataloader) * epochs
)

# Early stopping
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Model checkpointing
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pt')

# Loading
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```
