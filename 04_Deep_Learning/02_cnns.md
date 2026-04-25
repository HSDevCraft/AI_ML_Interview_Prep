# CNNs - Complete Guide

## ⚡ Interview Quick Summary

> **Core insight**: CNNs exploit spatial structure through local connectivity (convolutions) and translation invariance (weight sharing + pooling). Every design choice (kernel size, stride, padding, depth vs width) is a trade-off between receptive field, compute, and parameter count.

### Convolution Math — Must Know

```
Output size: O = floor((I - K + 2P) / S) + 1
  I = input size, K = kernel size, P = padding, S = stride

Example: I=32, K=3, P=1, S=1  →  O = (32-3+2)/1 + 1 = 32  (same padding)
Example: I=32, K=3, P=0, S=1  →  O = (32-3+0)/1 + 1 = 30  (valid padding)
Example: I=32, K=3, P=0, S=2  →  O = (32-3+0)/2 + 1 = 15  (stride 2 halves size)

Parameters in conv layer:
  Params = K×K×C_in×C_out + C_out (biases)
  Example: 3×3×3×64 + 64 = 1,792 params  (first AlexNet layer)
  vs Dense: 32×32×3×1000 = 3.07M params  → weight sharing is key!

Receptive field (grows with depth):
  Stack of three 3×3 convs has 7×7 receptive field
  Same as one 7×7 conv but fewer params: 3×(3×3×C×C) < 7×7×C×C
```

### Architecture Evolution — Key Ideas Per Model

| Model | Year | Key Innovation | Params |
|-------|------|---------------|--------|
| LeNet | 1989 | First CNN | 60K |
| AlexNet | 2012 | ReLU, Dropout, GPU | 60M |
| VGG | 2014 | Deep stacks of 3×3 convs | 138M |
| GoogLeNet | 2014 | Inception modules, 1×1 bottleneck | 6.8M |
| ResNet | 2015 | Residual connections, depth=152 | 25M |
| EfficientNet | 2019 | Compound scaling (depth+width+res) | 66M |
| ViT | 2020 | Pure attention (no convolution) | 86M |

### 🚨 Top Interview Pitfalls
- Forgetting **1×1 convolutions** (Inception): they change channel depth without affecting spatial dims; add non-linearity; cheap channel mixing
- Not knowing why ResNet uses **3×3+1×1 bottleneck**: saves compute (1×1 reduces channels before expensive 3×3)
- Saying "deeper = better" without mentioning the degradation problem ResNet solved
- Forgetting **global average pooling** replaces fully-connected layers in modern CNNs (fewer params, no fixed input size)

---

## Table of Contents
1. [Convolution Operations](#convolution-operations)
2. [CNN Building Blocks](#cnn-building-blocks)
3. [Classic Architectures](#classic-architectures)
4. [Modern Architectures](#modern-architectures)
5. [Object Detection](#object-detection)
6. [Image Segmentation](#image-segmentation)
7. [Interview Questions](#interview-questions)

---

## 1. Convolution Operations

### 2D Convolution

```python
import numpy as np

def conv2d(input, kernel, stride=1, padding=0):
    """
    2D Convolution operation.
    
    Input: (H, W) or (H, W, C)
    Kernel: (K_h, K_w) or (K_h, K_w, C_in, C_out)
    
    Output size: ((H + 2P - K) / S) + 1
    """
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


def conv2d_multichannel(input, kernels, bias, stride=1, padding=0):
    """
    Multi-channel convolution.
    
    Input: (H, W, C_in)
    Kernels: (K_h, K_w, C_in, C_out)
    Bias: (C_out,)
    Output: (H_out, W_out, C_out)
    """
    if padding > 0:
        input = np.pad(input, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    
    h_in, w_in, c_in = input.shape
    k_h, k_w, _, c_out = kernels.shape
    
    h_out = (h_in - k_h) // stride + 1
    w_out = (w_in - k_w) // stride + 1
    
    output = np.zeros((h_out, w_out, c_out))
    
    for i in range(h_out):
        for j in range(w_out):
            for k in range(c_out):
                region = input[i*stride:i*stride+k_h, j*stride:j*stride+k_w, :]
                output[i, j, k] = np.sum(region * kernels[:, :, :, k]) + bias[k]
    
    return output


# Output size formulas
def conv_output_size(input_size, kernel_size, stride=1, padding=0):
    """Calculate output spatial dimension."""
    return (input_size + 2 * padding - kernel_size) // stride + 1

def same_padding(input_size, kernel_size, stride=1):
    """Calculate padding needed for 'same' output size."""
    return ((stride - 1) * input_size - stride + kernel_size) // 2
```

### Types of Convolutions

```python
import torch
import torch.nn as nn

# Standard Convolution
conv_standard = nn.Conv2d(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    stride=1,
    padding=1  # 'same' padding
)
# Parameters: 64 * 128 * 3 * 3 + 128 = 73,856


# Depthwise Convolution
# Each input channel is convolved separately
depthwise = nn.Conv2d(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    groups=64  # groups = in_channels
)
# Parameters: 64 * 1 * 3 * 3 + 64 = 640


# Pointwise Convolution (1x1 Convolution)
# Changes number of channels without spatial mixing
pointwise = nn.Conv2d(
    in_channels=64,
    out_channels=128,
    kernel_size=1
)
# Parameters: 64 * 128 * 1 * 1 + 128 = 8,320


# Depthwise Separable Convolution (MobileNet)
# Factorizes standard conv into depthwise + pointwise
class DepthwiseSeparableConv(nn.Module):
    """
    Standard conv: C_in * C_out * K * K
    Separable conv: C_in * K * K + C_in * C_out
    
    Reduction factor: K² * C_out / (K² + C_out) ≈ K² for large C_out
    For K=3, C_out=128: ~9x fewer parameters
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# Dilated/Atrous Convolution
# Expands receptive field without increasing parameters
dilated = nn.Conv2d(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    padding=2,
    dilation=2  # Holes between kernel elements
)
# Effective kernel size: k + (k-1)*(d-1) = 3 + 2*1 = 5


# Transposed Convolution (Deconvolution)
# Upsamples spatial dimensions
transposed = nn.ConvTranspose2d(
    in_channels=64,
    out_channels=32,
    kernel_size=4,
    stride=2,
    padding=1
)
# Output size: (H-1)*stride - 2*padding + kernel_size


# Grouped Convolution
# Splits channels into groups, each processed separately
grouped = nn.Conv2d(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    padding=1,
    groups=4  # 4 groups of 16 channels each
)
# Parameters: 4 * (16 * 16 * 3 * 3) + 64 = 9,280 (vs 36,928 standard)
```

---

## 2. CNN Building Blocks

### Pooling Layers

```python
import numpy as np
import torch.nn as nn

def max_pool2d(input, pool_size=2, stride=2):
    """Max Pooling: Take maximum value in each region."""
    h_in, w_in = input.shape[:2]
    h_out = (h_in - pool_size) // stride + 1
    w_out = (w_in - pool_size) // stride + 1
    
    output = np.zeros((h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            region = input[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            output[i, j] = np.max(region)
    
    return output

def avg_pool2d(input, pool_size=2, stride=2):
    """Average Pooling: Take average value in each region."""
    h_in, w_in = input.shape[:2]
    h_out = (h_in - pool_size) // stride + 1
    w_out = (w_in - pool_size) // stride + 1
    
    output = np.zeros((h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            region = input[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            output[i, j] = np.mean(region)
    
    return output

def global_avg_pool(input):
    """
    Global Average Pooling: Average over entire spatial dimensions.
    Used to replace fully connected layers (fewer parameters).
    """
    return np.mean(input, axis=(0, 1))

# PyTorch pooling layers
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
adaptive_avg = nn.AdaptiveAvgPool2d((1, 1))  # Output size is 1x1
adaptive_max = nn.AdaptiveMaxPool2d((7, 7))  # Output size is 7x7


# Pooling comparison
"""
Max Pooling:
- Preserves dominant features
- Translation invariance
- Used in most classification networks

Average Pooling:
- Smoother, considers all values
- Good for localization tasks
- Used in final layer (Global Average Pooling)

Adaptive Pooling:
- Outputs fixed size regardless of input
- Useful for variable-size inputs
"""
```

### Normalization Layers

```python
import torch
import torch.nn as nn

# Batch Normalization (for CNNs)
class BatchNorm2d:
    """
    Normalizes across batch and spatial dimensions.
    For each channel: normalize (N, H, W) values.
    
    Position: Typically after conv, before activation
    """
    pass

bn = nn.BatchNorm2d(num_features=64)  # 64 channels
# Parameters: 2 * 64 = 128 (gamma, beta per channel)


# Layer Normalization
# Normalizes across channels and spatial dimensions
ln = nn.LayerNorm(normalized_shape=[64, 32, 32])  # C, H, W


# Instance Normalization
# Normalizes each sample, each channel independently
# Used in style transfer
instance_norm = nn.InstanceNorm2d(num_features=64)


# Group Normalization
# Divides channels into groups, normalizes within each group
# Works well with small batch sizes
group_norm = nn.GroupNorm(num_groups=8, num_channels=64)  # 8 channels per group


# Comparison
"""
BatchNorm: Best for large batches, standard choice for CNNs
LayerNorm: Batch-independent, used in Transformers
InstanceNorm: Sample and channel independent, style transfer
GroupNorm: Small batch alternative to BatchNorm

              Normalize over
BatchNorm:    (N, H, W)    for each channel
LayerNorm:    (C, H, W)    for each sample
InstanceNorm: (H, W)       for each sample, channel
GroupNorm:    (C/G, H, W)  for each sample, group
"""
```

### Common Blocks

```python
import torch
import torch.nn as nn

# Residual Block (ResNet)
class ResidualBlock(nn.Module):
    """
    x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU
    
    Skip connection allows gradient flow directly.
    Enables training very deep networks (100+ layers).
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Adjust dimensions for skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        return torch.relu(out)


# Bottleneck Block (ResNet-50+)
class Bottleneck(nn.Module):
    """
    1x1 (reduce) → 3x3 → 1x1 (expand)
    
    Reduces computation while maintaining representational power.
    """
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return torch.relu(out)


# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    """
    Channel attention: learns to weight channels.
    
    x → GlobalAvgPool → FC → ReLU → FC → Sigmoid → scale x
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# Inverted Residual Block (MobileNetV2)
class InvertedResidual(nn.Module):
    """
    Narrow → Wide → Narrow (inverted bottleneck)
    
    1x1 (expand) → 3x3 depthwise → 1x1 (project)
    Skip connection on narrow ends.
    """
    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super().__init__()
        hidden = in_channels * expansion
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        if expansion != 1:
            layers.append(nn.Conv2d(in_channels, hidden, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden))
            layers.append(nn.ReLU6(inplace=True))
        
        layers.extend([
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)
```

---

## 3. Classic Architectures

```python
import torch
import torch.nn as nn

# LeNet-5 (1998) - Handwritten digit recognition
class LeNet5(nn.Module):
    """
    First successful CNN. Used for MNIST.
    Conv → Pool → Conv → Pool → FC → FC → FC
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5),        # 28x28 → 24x24
            nn.Tanh(),
            nn.AvgPool2d(2),           # 24x24 → 12x12
            nn.Conv2d(6, 16, 5),       # 12x12 → 8x8
            nn.Tanh(),
            nn.AvgPool2d(2),           # 8x8 → 4x4
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# AlexNet (2012) - ImageNet breakthrough
class AlexNet(nn.Module):
    """
    Won ImageNet 2012. Introduced:
    - ReLU activation
    - Dropout regularization
    - Data augmentation
    - GPU training
    """
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        return self.classifier(x)


# VGG (2014) - Deeper is better
class VGG16(nn.Module):
    """
    Key insight: Stack 3x3 convolutions.
    Two 3x3 = one 5x5 receptive field, fewer parameters.
    """
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 5
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# GoogLeNet/Inception (2014) - Multi-scale features
class InceptionModule(nn.Module):
    """
    Process input with multiple filter sizes in parallel.
    Concatenate results along channel dimension.
    """
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, 
                 ch5x5_reduce, ch5x5, pool_proj):
        super().__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, 1),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, 5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, 1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)
```

---

## 4. Modern Architectures

```python
import torch
import torch.nn as nn

# ResNet (2015) - Skip connections
class ResNet(nn.Module):
    """
    Key insight: Skip connections solve vanishing gradient.
    Enables training 100+ layer networks.
    
    Variants: ResNet-18, 34, 50, 101, 152
    """
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = [block(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ResNet-18/34 use BasicBlock, ResNet-50+ use Bottleneck
def resnet18():
    return ResNet(ResidualBlock, [2, 2, 2, 2])

def resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


# DenseNet (2017) - Dense connections
class DenseBlock(nn.Module):
    """
    Each layer receives features from all preceding layers.
    Promotes feature reuse, reduces parameters.
    """
    def __init__(self, in_channels, growth_rate, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(self._make_layer(
                in_channels + i * growth_rate, growth_rate
            ))
    
    def _make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, 1, bias=False),  # Bottleneck
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False),
        )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        return torch.cat(features, dim=1)


# EfficientNet (2019) - Compound scaling
class MBConv(nn.Module):
    """
    Mobile Inverted Bottleneck with SE.
    Building block of EfficientNet.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super().__init__()
        hidden = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.SiLU(inplace=True),
            ])
        
        layers.extend([
            nn.Conv2d(hidden, hidden, kernel_size, stride, 
                     kernel_size // 2, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        ])
        
        # SE block
        layers.append(SEBlock(hidden))
        
        layers.extend([
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


# ConvNeXt (2022) - CNN competing with Transformers
class ConvNeXtBlock(nn.Module):
    """
    Modernized ResNet block inspired by Transformers.
    - Depthwise conv (7x7)
    - LayerNorm instead of BatchNorm
    - Inverted bottleneck (expand 4x)
    - GELU activation
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
    
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        return residual + x
```

---

## 5. Object Detection

```python
import torch
import torch.nn as nn
import numpy as np

# IoU (Intersection over Union)
def iou(box1, box2):
    """
    Calculate IoU between two boxes.
    Box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-6)


# GIoU (Generalized IoU) - better gradient for non-overlapping boxes
def giou(box1, box2):
    """GIoU = IoU - (C - Union) / C where C is smallest enclosing box."""
    iou_val = iou(box1, box2)
    
    # Enclosing box
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])
    
    c_area = (x2 - x1) * (y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - iou_val * (area1 + area2 - area1 * area2 / (area1 + area2 + 1e-6))
    
    return iou_val - (c_area - union) / (c_area + 1e-6)


# Non-Maximum Suppression
def nms(boxes, scores, iou_threshold=0.5):
    """
    Remove overlapping detections, keep highest scoring.
    
    boxes: (N, 4) array of [x1, y1, x2, y2]
    scores: (N,) array of confidence scores
    """
    indices = np.argsort(scores)[::-1]  # Sort by score descending
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        ious = np.array([iou(boxes[current], boxes[i]) for i in indices[1:]])
        
        # Keep boxes with IoU below threshold
        indices = indices[1:][ious < iou_threshold]
    
    return keep


# Soft-NMS
def soft_nms(boxes, scores, sigma=0.5, score_threshold=0.001):
    """
    Instead of removing overlapping boxes, reduce their scores.
    score = score * exp(-(iou^2) / sigma)
    """
    indices = list(range(len(boxes)))
    
    for i in range(len(boxes)):
        max_idx = max(indices, key=lambda j: scores[j])
        indices.remove(max_idx)
        
        for j in indices:
            iou_val = iou(boxes[max_idx], boxes[j])
            scores[j] *= np.exp(-(iou_val ** 2) / sigma)
    
    return [i for i in range(len(boxes)) if scores[i] > score_threshold]


# Anchor boxes
def generate_anchors(feature_map_size, scales, ratios, stride):
    """
    Generate anchor boxes for a feature map.
    
    scales: [32, 64, 128] - anchor sizes
    ratios: [0.5, 1.0, 2.0] - aspect ratios
    """
    anchors = []
    
    for y in range(feature_map_size):
        for x in range(feature_map_size):
            cx = (x + 0.5) * stride
            cy = (y + 0.5) * stride
            
            for scale in scales:
                for ratio in ratios:
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)
                    
                    anchors.append([
                        cx - w / 2,
                        cy - h / 2,
                        cx + w / 2,
                        cy + h / 2
                    ])
    
    return np.array(anchors)


# Detection architectures overview
"""
Two-Stage Detectors:
1. R-CNN (2014): Selective search → CNN → SVM
2. Fast R-CNN: Single CNN, ROI pooling
3. Faster R-CNN: Region Proposal Network (RPN)

Single-Stage Detectors:
1. YOLO: Single forward pass, grid-based
2. SSD: Multi-scale feature maps
3. RetinaNet: Focal loss for class imbalance

Modern Detectors:
1. DETR: Transformer-based, no anchors
2. YOLOv5-v8: Improved YOLO variants
"""
```

---

## 6. Image Segmentation

```python
import torch
import torch.nn as nn

# U-Net Architecture
class UNet(nn.Module):
    """
    Encoder-Decoder with skip connections.
    Used for semantic segmentation, especially medical imaging.
    """
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        
        # Encoder (contracting path)
        self.enc1 = self._conv_block(in_channels, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder (expanding path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self._conv_block(1024, 512)  # 512 + 512 from skip
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self._conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self._conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._conv_block(128, 64)
        
        self.out = nn.Conv2d(64, num_classes, 1)
        
        self.pool = nn.MaxPool2d(2)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        
        return self.out(d1)


# Segmentation types
"""
Semantic Segmentation: Pixel-wise class labels (no instance distinction)
- FCN, U-Net, DeepLab

Instance Segmentation: Detect + segment each object instance
- Mask R-CNN, YOLACT

Panoptic Segmentation: Semantic + Instance combined
- Panoptic FPN, MaskFormer
"""

# Segmentation loss functions
def dice_loss(pred, target, smooth=1e-6):
    """
    Dice Loss: 1 - 2*|A∩B| / (|A| + |B|)
    Good for imbalanced classes.
    """
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def focal_loss_segmentation(pred, target, gamma=2.0, alpha=0.25):
    """Focal loss for segmentation."""
    bce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce)
    focal = alpha * (1 - pt) ** gamma * bce
    return focal.mean()
```

---

## 7. Interview Questions

```python
# Q1: What is the difference between valid and same padding?
"""
Valid padding (no padding):
- Output size = (input - kernel) / stride + 1
- Output is smaller than input

Same padding:
- Output size = input size (when stride=1)
- Padding = (kernel - 1) / 2
- Preserves spatial dimensions
"""


# Q2: Why do CNNs use 3x3 kernels predominantly?
"""
1. Two 3x3 convs = one 5x5 receptive field
   Three 3x3 convs = one 7x7 receptive field

2. Fewer parameters: 2*(3*3) = 18 vs 5*5 = 25

3. More non-linearity (more ReLU activations)

4. More efficient on modern hardware

5. VGG paper showed this empirically
"""


# Q3: Explain 1x1 convolutions and their uses
"""
1x1 convolution:
- Changes number of channels
- No spatial mixing
- Acts as per-pixel fully connected layer

Uses:
1. Dimensionality reduction (bottleneck in Inception, ResNet)
2. Channel mixing after depthwise conv (MobileNet)
3. Increase model depth cheaply
4. Cross-channel interaction
"""


# Q4: Why do we need skip connections in ResNet?
"""
Problems without skip connections:
- Vanishing gradients in deep networks
- Degradation problem (deeper ≠ better)

Skip connections:
- Allow gradient to flow directly through identity mapping
- Enable training very deep networks (100+ layers)
- Learn residual F(x) = H(x) - x instead of H(x)
- Easier to learn F(x) = 0 (identity) than H(x) = x
"""


# Q5: What is the receptive field and how to calculate it?
"""
Receptive field: Region of input that affects one output pixel

Calculation (sequential layers):
RF_l = RF_{l-1} + (kernel_size - 1) * cumulative_stride

For 3 layers of 3x3 conv, stride 1:
RF = 1 + (3-1) + (3-1) + (3-1) = 7

Factors increasing RF:
- Larger kernels
- Strided convolutions
- Pooling layers
- Dilated convolutions
"""


# Q6: Why use Global Average Pooling instead of Flatten + FC?
"""
Flatten + FC:
- Many parameters (e.g., 512*7*7*4096 = 100M)
- Prone to overfitting
- Fixed input size required

Global Average Pooling:
- No learnable parameters
- Stronger regularization
- Works with any input size
- Directly maps features to classes
"""


# Q7: Explain depthwise separable convolution
"""
Standard conv: C_in * C_out * K * K operations

Depthwise separable:
1. Depthwise: Each channel convolved separately
   C_in * 1 * K * K
2. Pointwise: 1x1 conv to mix channels
   C_in * C_out * 1 * 1

Total: C_in * (K*K + C_out) vs C_in * C_out * K*K
Reduction: ~9x for 3x3, C_out=128
"""


# Q8: What is batch normalization doing mathematically?
"""
During training:
1. Compute batch mean and variance
2. Normalize: x_norm = (x - μ) / sqrt(σ² + ε)
3. Scale and shift: y = γ * x_norm + β

γ, β are learnable parameters.

Benefits:
- Reduces internal covariate shift
- Allows higher learning rates
- Reduces sensitivity to initialization
- Acts as regularization
"""


# Q9: How does feature pyramid network (FPN) work?
"""
FPN combines multi-scale features:

1. Bottom-up: Standard CNN (C2, C3, C4, C5)
2. Top-down: Upsample higher-level features
3. Lateral connections: 1x1 conv to match channels
4. Add top-down + lateral → P2, P3, P4, P5

Benefits:
- Rich semantics at all scales
- Better for detecting objects of various sizes
- Used in Faster R-CNN, Mask R-CNN
"""


# Q10: Compare YOLO vs Faster R-CNN
"""
Faster R-CNN (two-stage):
- Region Proposal Network → ROI pooling → classification
- More accurate, especially for small objects
- Slower (~7 FPS)
- Good for accuracy-critical applications

YOLO (single-stage):
- Single forward pass, grid-based prediction
- Much faster (~45-150+ FPS)
- Less accurate for small/overlapping objects
- Good for real-time applications

Trade-off: Speed vs Accuracy
"""
```
