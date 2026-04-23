# Neural Network Fundamentals - Complete Guide

## Table of Contents
1. [Perceptron and MLP](#perceptron-and-mlp)
2. [Activation Functions](#activation-functions)
3. [Loss Functions](#loss-functions)
4. [Optimizers](#optimizers)
5. [Weight Initialization](#weight-initialization)
6. [Regularization Techniques](#regularization-techniques)
7. [Backpropagation Deep Dive](#backpropagation-deep-dive)
8. [Interview Questions](#interview-questions)

---

## 1. Perceptron and MLP

### Single Perceptron

```python
import numpy as np

class Perceptron:
    """
    Single-layer perceptron (1957 - Rosenblatt)
    
    Model: y = activation(w·x + b)
    
    Limitations:
    - Can only learn linearly separable functions
    - Cannot solve XOR problem
    """
    
    def __init__(self, n_features, learning_rate=0.01):
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.lr = learning_rate
    
    def activation(self, x):
        """Step function."""
        return 1 if x >= 0 else 0
    
    def predict(self, X):
        linear_output = X @ self.weights + self.bias
        return np.array([self.activation(x) for x in linear_output])
    
    def fit(self, X, y, epochs=100):
        """Perceptron learning rule."""
        for epoch in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                prediction = self.activation(xi @ self.weights + self.bias)
                update = self.lr * (yi - prediction)
                
                # Update rule: w = w + lr * (y - y_hat) * x
                self.weights += update * xi
                self.bias += update
                
                if update != 0:
                    errors += 1
            
            if errors == 0:
                print(f"Converged at epoch {epoch}")
                break
        
        return self


# Perceptron convergence theorem:
# If data is linearly separable, perceptron will converge in finite steps
```

### Multi-Layer Perceptron (MLP)

```python
import numpy as np

class MLP:
    """
    Multi-Layer Perceptron from scratch.
    
    Universal Approximation Theorem:
    A neural network with one hidden layer can approximate any continuous function
    (given enough neurons).
    """
    
    def __init__(self, layer_sizes, activation='relu'):
        """
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
        """
        self.layers = []
        self.activation_name = activation
        
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            
            # He initialization for ReLU
            if activation == 'relu':
                w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            # Xavier initialization for tanh/sigmoid
            else:
                w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / (fan_in + fan_out))
            
            b = np.zeros((1, fan_out))
            self.layers.append({'w': w, 'b': b})
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """Forward pass through the network."""
        self.activations = [X]
        self.z_values = []
        
        for i, layer in enumerate(self.layers):
            z = self.activations[-1] @ layer['w'] + layer['b']
            self.z_values.append(z)
            
            # Output layer: softmax for classification
            if i == len(self.layers) - 1:
                a = self.softmax(z)
            else:
                # Hidden layers
                if self.activation_name == 'relu':
                    a = self.relu(z)
                elif self.activation_name == 'sigmoid':
                    a = self.sigmoid(z)
                else:
                    a = self.tanh(z)
            
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, y_true, learning_rate=0.01):
        """Backpropagation algorithm."""
        m = y_true.shape[0]
        n_layers = len(self.layers)
        
        # Output layer gradient (cross-entropy + softmax combined)
        dz = self.activations[-1] - y_true
        
        for i in reversed(range(n_layers)):
            # Gradient w.r.t weights and biases
            dw = (1 / m) * self.activations[i].T @ dz
            db = (1 / m) * np.sum(dz, axis=0, keepdims=True)
            
            if i > 0:
                # Gradient w.r.t previous layer's activation
                da = dz @ self.layers[i]['w'].T
                
                # Gradient through activation function
                if self.activation_name == 'relu':
                    dz = da * self.relu_derivative(self.z_values[i - 1])
                elif self.activation_name == 'sigmoid':
                    dz = da * self.sigmoid_derivative(self.z_values[i - 1])
                else:
                    dz = da * self.tanh_derivative(self.z_values[i - 1])
            
            # Update weights
            self.layers[i]['w'] -= learning_rate * dw
            self.layers[i]['b'] -= learning_rate * db
    
    def fit(self, X, y, epochs=1000, learning_rate=0.01, verbose=True):
        """Train the network."""
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = -np.mean(np.sum(y * np.log(y_pred + 1e-15), axis=1))
            
            # Backward pass
            self.backward(y, learning_rate)
            
            if verbose and epoch % 100 == 0:
                accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return self
    
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)


# Example: XOR problem (requires hidden layer)
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # One-hot encoded

mlp = MLP([2, 4, 2], activation='relu')
mlp.fit(X_xor, y_xor, epochs=1000, learning_rate=0.1)
```

---

## 2. Activation Functions

### Comparison of Activation Functions

```python
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid
def sigmoid(x):
    """
    σ(x) = 1 / (1 + e^(-x))
    
    Range: (0, 1)
    
    Pros:
    - Smooth, differentiable
    - Output bounded (good for probabilities)
    
    Cons:
    - Vanishing gradient (saturates for large |x|)
    - Not zero-centered (outputs always positive)
    - Expensive to compute (exp)
    
    Use: Output layer for binary classification
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)  # Max derivative = 0.25 at x=0


# Tanh
def tanh(x):
    """
    tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    
    Range: (-1, 1)
    
    Pros:
    - Zero-centered
    - Stronger gradients than sigmoid
    
    Cons:
    - Still has vanishing gradient problem
    
    Use: RNN hidden states (historically)
    """
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2  # Max = 1 at x=0


# ReLU (Rectified Linear Unit)
def relu(x):
    """
    ReLU(x) = max(0, x)
    
    Range: [0, ∞)
    
    Pros:
    - No vanishing gradient for x > 0
    - Computationally efficient
    - Sparse activation (biological plausibility)
    - Faster convergence
    
    Cons:
    - Dying ReLU: neurons can "die" (always output 0)
    - Not zero-centered
    - Unbounded (can explode)
    
    Use: Default for hidden layers in CNNs and MLPs
    """
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)


# Leaky ReLU
def leaky_relu(x, alpha=0.01):
    """
    LeakyReLU(x) = x if x > 0, else αx
    
    Fixes dying ReLU problem by allowing small gradient for x < 0.
    
    Parametric ReLU (PReLU): α is learned parameter
    """
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


# ELU (Exponential Linear Unit)
def elu(x, alpha=1.0):
    """
    ELU(x) = x if x > 0, else α(e^x - 1)
    
    Pros:
    - Smooth, negative values push mean toward zero
    - More robust to noise
    
    Cons:
    - Slower than ReLU (exp computation)
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


# SELU (Scaled ELU)
def selu(x, alpha=1.6732, scale=1.0507):
    """
    SELU(x) = scale * (x if x > 0, else α(e^x - 1))
    
    Self-normalizing: maintains mean≈0 and var≈1 through layers.
    Requires specific initialization (lecun_normal) and architecture.
    """
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))


# GELU (Gaussian Error Linear Unit)
def gelu(x):
    """
    GELU(x) = x * Φ(x) where Φ is CDF of standard normal
    
    Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    
    Used in: BERT, GPT, modern transformers
    
    Pros:
    - Smooth approximation of ReLU
    - Allows small gradients for negative inputs
    - State-of-the-art performance in NLP
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# Swish / SiLU
def swish(x, beta=1.0):
    """
    Swish(x) = x * sigmoid(βx)
    
    SiLU (Sigmoid Linear Unit) is Swish with β=1
    
    Used in: EfficientNet, modern vision models
    
    Pros:
    - Self-gated (multiplicative interaction)
    - Smooth, non-monotonic
    - Outperforms ReLU in many tasks
    """
    return x * sigmoid(beta * x)


# Mish
def mish(x):
    """
    Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
    
    Used in: YOLOv4, modern CNNs
    
    Similar benefits to Swish, slightly smoother.
    """
    return x * np.tanh(np.log1p(np.exp(x)))


# Softmax (for multi-class output)
def softmax(x):
    """
    softmax(x)_i = e^(x_i) / Σ_j e^(x_j)
    
    Converts logits to probabilities (sum to 1).
    Used in output layer for multi-class classification.
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# Summary: When to use what
"""
Hidden Layers:
- Default: ReLU (fast, works well)
- If dying ReLU: Leaky ReLU, ELU, or GELU
- Modern NLP: GELU
- Modern Vision: Swish/SiLU or Mish

Output Layer:
- Binary classification: Sigmoid
- Multi-class classification: Softmax
- Regression: Linear (no activation)
- Regression with bounds: Sigmoid scaled to range
"""
```

---

## 3. Loss Functions

### Classification Losses

```python
import numpy as np

def binary_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    BCE = -1/n Σ [y·log(p) + (1-y)·log(1-p)]
    
    Used for: Binary classification (sigmoid output)
    
    Gradient: ∂L/∂p = -y/p + (1-y)/(1-p)
    Combined with sigmoid: ∂L/∂z = p - y (simple!)
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    CCE = -Σ_i y_i · log(p_i)
    
    Used for: Multi-class classification (softmax output)
    y_true: One-hot encoded
    
    Combined with softmax: ∂L/∂z = p - y
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def sparse_categorical_cross_entropy(y_true, y_pred, epsilon=1e-15):
    """
    Same as CCE but y_true is class indices instead of one-hot.
    More memory efficient.
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    n_samples = y_true.shape[0]
    return -np.mean(np.log(y_pred[np.arange(n_samples), y_true]))


def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25, epsilon=1e-15):
    """
    FL = -α(1-p)^γ · log(p)
    
    Used for: Imbalanced classification
    
    γ (gamma): Focusing parameter
    - γ=0: Same as cross-entropy
    - γ>0: Down-weights easy examples, focuses on hard ones
    
    α (alpha): Class weight for positive class
    
    From RetinaNet paper (Lin et al., 2017)
    """
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # p_t = p for y=1, (1-p) for y=0
    pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    
    # Focal weight
    focal_weight = (1 - pt) ** gamma
    
    # Alpha weight
    alpha_weight = y_true * alpha + (1 - y_true) * (1 - alpha)
    
    return -np.mean(alpha_weight * focal_weight * np.log(pt))


def label_smoothing_loss(y_true, y_pred, smoothing=0.1, epsilon=1e-15):
    """
    Label smoothing: y_smooth = y * (1 - ε) + ε / K
    
    Prevents overconfident predictions.
    Used in modern image classification.
    """
    n_classes = y_true.shape[1]
    y_smooth = y_true * (1 - smoothing) + smoothing / n_classes
    
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_smooth * np.log(y_pred), axis=1))
```

### Regression Losses

```python
def mse_loss(y_true, y_pred):
    """
    MSE = 1/n Σ (y - ŷ)²
    
    Pros: Penalizes large errors heavily
    Cons: Sensitive to outliers
    Gradient: 2(ŷ - y)
    """
    return np.mean((y_true - y_pred) ** 2)


def mae_loss(y_true, y_pred):
    """
    MAE = 1/n Σ |y - ŷ|
    
    Pros: Robust to outliers
    Cons: Gradient is constant (±1), can be slow near optimum
    """
    return np.mean(np.abs(y_true - y_pred))


def huber_loss(y_true, y_pred, delta=1.0):
    """
    Huber Loss = 0.5 * (y - ŷ)² if |y - ŷ| ≤ δ
                 δ * (|y - ŷ| - 0.5δ) otherwise
    
    Combines MSE (for small errors) and MAE (for large errors).
    Used in robust regression and reinforcement learning.
    """
    error = y_true - y_pred
    is_small = np.abs(error) <= delta
    
    small_loss = 0.5 * error ** 2
    large_loss = delta * (np.abs(error) - 0.5 * delta)
    
    return np.mean(np.where(is_small, small_loss, large_loss))


def log_cosh_loss(y_true, y_pred):
    """
    Log-Cosh = 1/n Σ log(cosh(y - ŷ))
    
    Similar to Huber but smoother (twice differentiable).
    """
    error = y_true - y_pred
    return np.mean(np.log(np.cosh(error)))


def quantile_loss(y_true, y_pred, quantile=0.5):
    """
    Quantile Loss = Σ max(q*(y-ŷ), (q-1)*(y-ŷ))
    
    q=0.5: Median regression (equivalent to MAE)
    q<0.5: Predict lower than median
    q>0.5: Predict higher than median
    
    Used for prediction intervals.
    """
    error = y_true - y_pred
    return np.mean(np.maximum(quantile * error, (quantile - 1) * error))
```

### Contrastive and Other Losses

```python
def contrastive_loss(y_true, distance, margin=1.0):
    """
    Contrastive Loss (Siamese Networks)
    L = y * d² + (1-y) * max(0, margin - d)²
    
    y=1: Similar pairs (minimize distance)
    y=0: Dissimilar pairs (maximize distance up to margin)
    """
    similar_loss = y_true * distance ** 2
    dissimilar_loss = (1 - y_true) * np.maximum(0, margin - distance) ** 2
    return np.mean(similar_loss + dissimilar_loss)


def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Triplet Loss
    L = max(0, d(a, p) - d(a, n) + margin)
    
    Push positive closer to anchor, negative farther.
    Used in face recognition (FaceNet).
    """
    pos_dist = np.sum((anchor - positive) ** 2, axis=1)
    neg_dist = np.sum((anchor - negative) ** 2, axis=1)
    return np.mean(np.maximum(0, pos_dist - neg_dist + margin))


def kl_divergence(p, q, epsilon=1e-15):
    """
    KL(P || Q) = Σ p_i * log(p_i / q_i)
    
    Measures difference between distributions.
    Used in VAEs, knowledge distillation.
    """
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return np.sum(p * np.log(p / q), axis=-1).mean()
```

---

## 4. Optimizers

### Gradient Descent Variants

```python
import numpy as np

class SGD:
    """
    Stochastic Gradient Descent with Momentum and Nesterov
    
    Update rule:
    v_t = β * v_{t-1} + (1 - β) * g_t  (or just g_t without dampening)
    θ_t = θ_{t-1} - lr * v_t
    
    Nesterov: Compute gradient at θ - β*v (look ahead)
    """
    
    def __init__(self, lr=0.01, momentum=0.0, nesterov=False, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.velocity = None
    
    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        
        for i, (p, g) in enumerate(zip(params, grads)):
            # L2 regularization (weight decay)
            if self.weight_decay > 0:
                g = g + self.weight_decay * p
            
            if self.momentum > 0:
                self.velocity[i] = self.momentum * self.velocity[i] + g
                
                if self.nesterov:
                    g = g + self.momentum * self.velocity[i]
                else:
                    g = self.velocity[i]
            
            p -= self.lr * g


class RMSprop:
    """
    RMSprop: Adaptive learning rate per parameter
    
    s_t = β * s_{t-1} + (1-β) * g_t²
    θ_t = θ_{t-1} - lr * g_t / (sqrt(s_t) + ε)
    
    Adapts learning rate based on recent gradient magnitudes.
    Good for RNNs and non-stationary objectives.
    """
    
    def __init__(self, lr=0.001, beta=0.99, epsilon=1e-8):
        self.lr = lr
        self.beta = beta
        self.epsilon = epsilon
        self.s = None
    
    def step(self, params, grads):
        if self.s is None:
            self.s = [np.zeros_like(p) for p in params]
        
        for i, (p, g) in enumerate(zip(params, grads)):
            self.s[i] = self.beta * self.s[i] + (1 - self.beta) * g ** 2
            p -= self.lr * g / (np.sqrt(self.s[i]) + self.epsilon)


class Adam:
    """
    Adam: Adaptive Moment Estimation
    
    m_t = β₁ * m_{t-1} + (1-β₁) * g_t       (first moment - momentum)
    v_t = β₂ * v_{t-1} + (1-β₂) * g_t²      (second moment - RMSprop)
    m̂_t = m_t / (1 - β₁^t)                  (bias correction)
    v̂_t = v_t / (1 - β₂^t)                  (bias correction)
    θ_t = θ_{t-1} - lr * m̂_t / (sqrt(v̂_t) + ε)
    
    Combines momentum and adaptive learning rates.
    Default choice for most deep learning tasks.
    """
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        
        for i, (p, g) in enumerate(zip(params, grads)):
            # Update biased moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class AdamW:
    """
    AdamW: Adam with Decoupled Weight Decay
    
    Key insight: L2 regularization ≠ weight decay in adaptive optimizers
    
    Standard Adam with L2:
    g_t = ∇L + λθ
    
    AdamW (decoupled):
    θ_t = θ_{t-1} - lr * (m̂_t / (sqrt(v̂_t) + ε) + λ * θ_{t-1})
    
    Used in most modern transformers (BERT, GPT, etc.)
    """
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0
    
    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        
        for i, (p, g) in enumerate(zip(params, grads)):
            # Decoupled weight decay
            p -= self.lr * self.weight_decay * p
            
            # Adam update
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class LAMB:
    """
    LAMB: Layer-wise Adaptive Moments for Batch training
    
    Adds layer-wise learning rate scaling to Adam.
    Enables training with very large batch sizes.
    Used for training BERT in 76 minutes.
    """
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-6, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
        self.t = 0
    
    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        
        for i, (p, g) in enumerate(zip(params, grads)):
            # Adam moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update direction with weight decay
            update = m_hat / (np.sqrt(v_hat) + self.epsilon) + self.weight_decay * p
            
            # Layer-wise scaling (trust ratio)
            p_norm = np.linalg.norm(p)
            u_norm = np.linalg.norm(update)
            
            if p_norm > 0 and u_norm > 0:
                trust_ratio = p_norm / u_norm
            else:
                trust_ratio = 1.0
            
            p -= self.lr * trust_ratio * update
```

### Learning Rate Schedules

```python
class LearningRateScheduler:
    """Common learning rate schedules."""
    
    @staticmethod
    def step_decay(initial_lr, epoch, drop_rate=0.5, epochs_drop=10):
        """Drop by factor every N epochs."""
        return initial_lr * (drop_rate ** (epoch // epochs_drop))
    
    @staticmethod
    def exponential_decay(initial_lr, epoch, decay_rate=0.95):
        """Exponential decay: lr = lr_0 * γ^epoch"""
        return initial_lr * (decay_rate ** epoch)
    
    @staticmethod
    def cosine_annealing(initial_lr, epoch, total_epochs, min_lr=0):
        """Cosine annealing: smooth decay to min_lr."""
        return min_lr + 0.5 * (initial_lr - min_lr) * (
            1 + np.cos(np.pi * epoch / total_epochs)
        )
    
    @staticmethod
    def warmup_cosine(initial_lr, epoch, warmup_epochs, total_epochs, min_lr=0):
        """Linear warmup followed by cosine decay."""
        if epoch < warmup_epochs:
            return initial_lr * epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(np.pi * progress))
    
    @staticmethod
    def one_cycle(max_lr, epoch, total_epochs, pct_start=0.3, div_factor=25, final_div=1e4):
        """One Cycle Policy: warmup → anneal → final anneal."""
        initial_lr = max_lr / div_factor
        min_lr = max_lr / final_div
        
        if epoch < total_epochs * pct_start:
            # Warmup phase
            progress = epoch / (total_epochs * pct_start)
            return initial_lr + (max_lr - initial_lr) * progress
        else:
            # Annealing phase
            progress = (epoch - total_epochs * pct_start) / (total_epochs * (1 - pct_start))
            return max_lr - (max_lr - min_lr) * progress
```

---

## 5. Weight Initialization

```python
import numpy as np

def zeros_init(shape):
    """All zeros - BAD! All neurons compute same thing."""
    return np.zeros(shape)

def random_init(shape, scale=0.01):
    """Small random - can lead to vanishing/exploding gradients."""
    return np.random.randn(*shape) * scale

def xavier_uniform(fan_in, fan_out):
    """
    Xavier/Glorot Uniform Initialization
    
    W ~ U[-a, a] where a = sqrt(6 / (fan_in + fan_out))
    
    Maintains variance of activations and gradients.
    Best for: Sigmoid, Tanh activations
    """
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

def xavier_normal(fan_in, fan_out):
    """
    Xavier/Glorot Normal Initialization
    
    W ~ N(0, σ²) where σ = sqrt(2 / (fan_in + fan_out))
    """
    std = np.sqrt(2 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out) * std

def he_uniform(fan_in, fan_out):
    """
    He/Kaiming Uniform Initialization
    
    W ~ U[-a, a] where a = sqrt(6 / fan_in)
    
    Accounts for ReLU zeroing out half the neurons.
    Best for: ReLU, Leaky ReLU activations
    """
    limit = np.sqrt(6 / fan_in)
    return np.random.uniform(-limit, limit, (fan_in, fan_out))

def he_normal(fan_in, fan_out):
    """
    He/Kaiming Normal Initialization
    
    W ~ N(0, σ²) where σ = sqrt(2 / fan_in)
    """
    std = np.sqrt(2 / fan_in)
    return np.random.randn(fan_in, fan_out) * std

def lecun_normal(fan_in, fan_out):
    """
    LeCun Normal Initialization
    
    W ~ N(0, σ²) where σ = sqrt(1 / fan_in)
    
    Best for: SELU activation
    """
    std = np.sqrt(1 / fan_in)
    return np.random.randn(fan_in, fan_out) * std

def orthogonal_init(shape, gain=1.0):
    """
    Orthogonal Initialization
    
    Initializes weight matrix as orthogonal matrix.
    Preserves gradient norm during backprop.
    Good for: RNNs
    """
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.randn(*flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return gain * q[:shape[0], :shape[1]]


# Summary
"""
Activation      | Initialization
----------------|----------------
Sigmoid/Tanh    | Xavier (Glorot)
ReLU/LeakyReLU  | He (Kaiming)
SELU            | LeCun
RNN             | Orthogonal
"""
```

---

## 6. Regularization Techniques

```python
import numpy as np

# L1 Regularization (Lasso)
def l1_regularization(weights, lambda_):
    """
    L1 = λ Σ |w|
    
    Gradient: λ * sign(w)
    Effect: Promotes sparsity (many weights → 0)
    """
    return lambda_ * np.sum(np.abs(weights))

def l1_gradient(weights, lambda_):
    return lambda_ * np.sign(weights)


# L2 Regularization (Ridge / Weight Decay)
def l2_regularization(weights, lambda_):
    """
    L2 = λ Σ w²
    
    Gradient: 2λw
    Effect: Penalizes large weights, prevents overfitting
    """
    return lambda_ * np.sum(weights ** 2)

def l2_gradient(weights, lambda_):
    return 2 * lambda_ * weights


# Dropout
class Dropout:
    """
    Dropout: Randomly zero out neurons during training.
    
    Training: x_out = x * mask / (1 - p)  [inverted dropout]
    Inference: x_out = x
    
    Effect:
    - Prevents co-adaptation of neurons
    - Implicit ensemble of sub-networks
    - Regularization without changing architecture
    
    Typical p: 0.5 for hidden, 0.2 for input
    """
    
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
    
    def forward(self, x, training=True):
        if not training or self.p == 0:
            return x
        
        self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
        return x * self.mask
    
    def backward(self, grad):
        return grad * self.mask


# DropConnect
class DropConnect:
    """
    DropConnect: Drop connections (weights) instead of neurons.
    More fine-grained than Dropout.
    """
    
    def __init__(self, p=0.5):
        self.p = p
    
    def forward(self, x, w, training=True):
        if not training:
            return x @ w
        
        mask = (np.random.rand(*w.shape) > self.p) / (1 - self.p)
        return x @ (w * mask)


# Batch Normalization
class BatchNorm:
    """
    BatchNorm: Normalize activations within each mini-batch.
    
    Training:
    μ = mean(x, axis=batch)
    σ² = var(x, axis=batch)
    x̂ = (x - μ) / sqrt(σ² + ε)
    y = γx̂ + β
    
    Also maintains running mean/var for inference.
    
    Benefits:
    - Reduces internal covariate shift
    - Allows higher learning rates
    - Acts as regularizer
    - Reduces sensitivity to initialization
    
    Position: Usually after linear/conv, before activation
    """
    
    def __init__(self, num_features, momentum=0.1, epsilon=1e-5):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.momentum = momentum
        self.epsilon = epsilon
        
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x, training=True):
        if training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            
            # Cache for backward
            self.x_centered = x - mean
            self.std = np.sqrt(var + self.epsilon)
            self.x_norm = self.x_centered / self.std
        else:
            self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        return self.gamma * self.x_norm + self.beta
    
    def backward(self, grad_out):
        n = grad_out.shape[0]
        
        # Gradients for gamma and beta
        self.grad_gamma = np.sum(grad_out * self.x_norm, axis=0)
        self.grad_beta = np.sum(grad_out, axis=0)
        
        # Gradient for input
        dx_norm = grad_out * self.gamma
        dvar = np.sum(dx_norm * self.x_centered * -0.5 * self.std**-3, axis=0)
        dmean = np.sum(dx_norm * -1 / self.std, axis=0) + dvar * np.mean(-2 * self.x_centered, axis=0)
        
        return dx_norm / self.std + dvar * 2 * self.x_centered / n + dmean / n


# Layer Normalization
class LayerNorm:
    """
    LayerNorm: Normalize across features (not batch).
    
    Used in Transformers (batch-independent).
    
    Normalizes each sample independently:
    μ = mean(x, axis=features)
    σ² = var(x, axis=features)
    """
    
    def __init__(self, normalized_shape, epsilon=1e-5):
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
        self.epsilon = epsilon
    
    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        
        self.x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        return self.gamma * self.x_norm + self.beta


# Data Augmentation (implicit regularization)
"""
Image augmentation:
- Random crop, flip, rotation
- Color jittering
- Mixup, CutMix, CutOut

Text augmentation:
- Back-translation
- Synonym replacement
- Random deletion/insertion

Audio augmentation:
- Time stretch, pitch shift
- SpecAugment
"""


# Early Stopping
class EarlyStopping:
    """
    Stop training when validation loss stops improving.
    """
    
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.should_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop
```

---

## 7. Backpropagation Deep Dive

```python
"""
Backpropagation: Efficient computation of gradients using chain rule.

For loss L and network f(x) = f_n(f_{n-1}(...f_1(x)...)):

∂L/∂θ_i = ∂L/∂f_n · ∂f_n/∂f_{n-1} · ... · ∂f_{i+1}/∂f_i · ∂f_i/∂θ_i

Forward pass: Compute and store intermediate values
Backward pass: Propagate gradients from output to input
"""

# Example: Two-layer network step by step
def backprop_example(X, y, W1, b1, W2, b2):
    """
    Network: y = softmax(ReLU(X @ W1 + b1) @ W2 + b2)
    Loss: Cross-entropy
    """
    # === FORWARD PASS ===
    # Layer 1
    z1 = X @ W1 + b1          # (batch, hidden)
    a1 = np.maximum(0, z1)    # ReLU
    
    # Layer 2 (output)
    z2 = a1 @ W2 + b2         # (batch, output)
    exp_z2 = np.exp(z2 - z2.max(axis=1, keepdims=True))
    a2 = exp_z2 / exp_z2.sum(axis=1, keepdims=True)  # Softmax
    
    # Loss (cross-entropy)
    m = X.shape[0]
    loss = -np.sum(y * np.log(a2 + 1e-15)) / m
    
    # === BACKWARD PASS ===
    # Output layer gradient (softmax + cross-entropy combined)
    # ∂L/∂z2 = a2 - y
    dz2 = (a2 - y) / m
    
    # Gradients for W2, b2
    # ∂L/∂W2 = a1^T @ ∂L/∂z2
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)
    
    # Gradient for a1
    # ∂L/∂a1 = ∂L/∂z2 @ W2^T
    da1 = dz2 @ W2.T
    
    # Gradient through ReLU
    # ∂L/∂z1 = ∂L/∂a1 * ReLU'(z1)
    dz1 = da1 * (z1 > 0).astype(float)
    
    # Gradients for W1, b1
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)
    
    return loss, dW1, db1, dW2, db2


# Gradient checking (verify implementation)
def gradient_check(f, x, epsilon=1e-7):
    """
    Numerical gradient: (f(x+ε) - f(x-ε)) / 2ε
    Compare with analytical gradient.
    """
    numerical_grad = np.zeros_like(x)
    
    for i in range(x.size):
        x_plus = x.copy()
        x_plus.flat[i] += epsilon
        x_minus = x.copy()
        x_minus.flat[i] -= epsilon
        
        numerical_grad.flat[i] = (f(x_plus) - f(x_minus)) / (2 * epsilon)
    
    return numerical_grad
```

---

## 8. Interview Questions

```python
# Q1: Why do we need non-linear activation functions?
"""
Without non-linearity:
- Multiple linear layers collapse to single linear transformation
- Network can only learn linear functions
- Cannot solve XOR or any non-linear problem

Non-linearity enables:
- Learning complex decision boundaries
- Universal approximation theorem
- Hierarchical feature learning
"""


# Q2: What is vanishing gradient and how to address it?
"""
Problem: Gradients become very small in deep networks.
- Sigmoid/tanh saturate for large |x| → gradient ≈ 0
- Gradients multiply through layers → exponentially small

Solutions:
1. ReLU and variants (gradient = 1 for x > 0)
2. Skip connections (ResNet) - gradient highway
3. Proper initialization (He, Xavier)
4. Batch Normalization - keeps activations in good range
5. LSTM/GRU for RNNs - gating mechanism
"""


# Q3: What is exploding gradient and how to address it?
"""
Problem: Gradients become very large → unstable training.
- Common in RNNs
- Weights update too much

Solutions:
1. Gradient clipping: clip ||g|| to max value
2. Proper initialization
3. Batch Normalization
4. Lower learning rate
5. LSTM/GRU (controlled information flow)
"""


# Q4: Adam vs SGD - when to use which?
"""
Adam:
- Converges faster initially
- Less hyperparameter tuning
- Good default choice
- May generalize slightly worse

SGD with momentum:
- Often better final performance
- Better generalization
- Requires more tuning (LR schedule)
- Used for SOTA image classification

Recommendation:
- Start with Adam for prototyping
- Use SGD for final training if performance matters
"""


# Q5: Why is weight initialization important?
"""
Bad initialization leads to:
- Vanishing gradients (weights too small)
- Exploding gradients (weights too large)
- Symmetry breaking failure (all same)
- Slow convergence

Good initialization:
- Maintains variance of activations across layers
- Allows gradient flow
- Xavier for sigmoid/tanh, He for ReLU
"""


# Q6: What is the difference between BatchNorm and LayerNorm?
"""
BatchNorm:
- Normalizes across batch dimension
- μ, σ computed over (batch, height, width)
- Depends on batch size
- Used in CNNs

LayerNorm:
- Normalizes across feature dimension
- μ, σ computed over (features)
- Batch-independent
- Used in Transformers, RNNs
"""


# Q7: How does Dropout work during training vs inference?
"""
Training:
- Randomly zero out neurons with probability p
- Scale remaining by 1/(1-p) [inverted dropout]
- Creates ensemble of sub-networks

Inference:
- Use all neurons
- No scaling needed (already done in training)

Effect: Regularization, prevents co-adaptation
"""


# Q8: What is the dying ReLU problem?
"""
Problem: Neurons output 0 for all inputs.
- If z < 0, gradient = 0, no learning
- Once dead, stays dead

Solutions:
- Leaky ReLU: small gradient for x < 0
- ELU, SELU: smooth negative region
- Proper initialization
- Lower learning rate
- PReLU: learnable negative slope
"""


# Q9: Explain the difference between L1 and L2 regularization
"""
L1 (Lasso): λΣ|w|
- Sparse solutions (many w = 0)
- Feature selection
- Gradient = λ·sign(w)

L2 (Ridge): λΣw²
- Small but non-zero weights
- Smooth solution
- Gradient = 2λw

L1 + L2 (Elastic Net): Best of both
"""


# Q10: What is gradient clipping and why use it?
"""
Gradient Clipping: Limit gradient magnitude

Methods:
1. Clip by value: g = clip(g, -max, max)
2. Clip by norm: if ||g|| > max: g = g * max / ||g||

Why:
- Prevents exploding gradients
- Stabilizes training, especially for RNNs
- Allows higher learning rates

Typical value: 1.0 or 5.0
"""
```
