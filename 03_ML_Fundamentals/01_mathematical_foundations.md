# Mathematical Foundations for Machine Learning - Complete Guide

## ⚡ Interview Quick Summary

> **Core insight**: ML math is linear algebra + calculus + probability. Interviewers test intuition ("what does the Hessian tell you?") more than derivation. Know the formulas AND what they mean geometrically.

### Essential Formulas — Must Know by Heart

```
Linear Algebra:
  Matrix multiply: (m×n) × (n×p) = (m×p)  [inner dims must match]
  Eigendecomposition: A = VΛV^{-1}  [A symmetric: V is orthogonal, Λ is diagonal of eigenvalues]
  SVD: A = UΣV^T  [U: left singular vecs, Σ: singular values, V: right singular vecs]
  Trace: tr(A) = sum of diagonal = sum of eigenvalues
  Determinant: det(A) = product of eigenvalues; det=0 ⇒ matrix is singular (not invertible)

Calculus:
  Chain rule: df/dx = df/dg · dg/dx  [foundation of backprop]
  Gradient: ∇f = vector of partial derivatives → points in direction of steepest ascent
  Jacobian: matrix of all first partial derivatives (for vector-valued functions)
  Hessian: matrix of all second partial derivatives; PSD Hessian ⇒ convex function

Probability:
  Bayes: P(A|B) = P(B|A)P(A) / P(B)  [posterior ∝ likelihood × prior]
  Expectation: E[X] = Σ x P(X=x)
  Variance: Var[X] = E[X²] - (E[X])²
  Normal: N(μ,σ²)   P(x) = exp(-(x-μ)²/2σ²) / sqrt(2πσ²)
  MLE: θ* = argmax Σ log P(x_i | θ)  [maximize log-likelihood]
  MAP: θ* = argmax Σ log P(x_i|θ) + log P(θ)  [add log-prior = regularization!]
```

### Geometric Intuitions — For Interview Explanations

| Concept | Geometric meaning |
|---------|------------------|
| Eigenvectors | Directions the matrix scales (doesn't rotate) |
| Gradient | Arrow pointing uphill, perpendicular to level sets |
| Hessian eigenvalues | Curvature in each direction; all positive = bowl (convex) |
| Covariance matrix | Shape and orientation of data cloud |
| L2 norm of weights | Length of weight vector in parameter space |
| Dot product = 0 | Orthogonal (perpendicular) vectors |

### 🚨 Top Interview Pitfalls
- Confusing **MLE** (maximize likelihood) with **MAP** (likelihood + prior) — L2 regularization IS a Gaussian prior; L1 IS a Laplace prior
- Not knowing that **PCA = SVD** on centered data — top-k singular vectors are the principal components
- Saying "gradient points toward minimum" — it points AWAY from minimum (toward steepest ascent); we go NEGATIVE gradient for descent
- Not knowing that **covariance matrix PSD** means Var[a^T X] ≥ 0 for any vector a (all variances are non-negative)

---

## Table of Contents
1. [Linear Algebra](#linear-algebra)
2. [Calculus](#calculus)
3. [Probability Theory](#probability-theory)
4. [Statistics](#statistics)
5. [Optimization](#optimization)
6. [Interview Questions](#interview-questions)

---

## 1. Linear Algebra

### Vectors

```python
import numpy as np

# Vector: ordered array of numbers
v = np.array([1, 2, 3])

# Vector operations
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Addition/Subtraction
v_add = v1 + v2  # [5, 7, 9]
v_sub = v1 - v2  # [-3, -3, -3]

# Scalar multiplication
v_scaled = 2 * v1  # [2, 4, 6]

# Dot product (inner product)
# Sum of element-wise products
dot = np.dot(v1, v2)  # 1*4 + 2*5 + 3*6 = 32
dot = v1 @ v2         # Same result

# Geometric interpretation:
# dot(a, b) = ||a|| * ||b|| * cos(θ)
# If dot = 0, vectors are orthogonal (perpendicular)

# Cross product (3D only)
cross = np.cross(v1, v2)  # [-3, 6, -3]

# Vector norms (magnitude/length)
l1_norm = np.linalg.norm(v1, ord=1)   # Manhattan: |1|+|2|+|3| = 6
l2_norm = np.linalg.norm(v1, ord=2)   # Euclidean: sqrt(1²+2²+3²) = 3.74
linf_norm = np.linalg.norm(v1, ord=np.inf)  # Max: max(|1|,|2|,|3|) = 3

# Unit vector (normalized)
unit_v = v1 / np.linalg.norm(v1)
# ||unit_v|| = 1

# Projection of v1 onto v2
proj = (np.dot(v1, v2) / np.dot(v2, v2)) * v2
```

### Matrices

```python
# Matrix: 2D array of numbers
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix operations
C = A + B           # Element-wise addition
D = A - B           # Element-wise subtraction
E = A * B           # Element-wise multiplication (Hadamard)
F = A @ B           # Matrix multiplication
G = np.matmul(A, B) # Same as @

# Matrix multiplication rules:
# (m x n) @ (n x p) = (m x p)
# NOT commutative: A @ B ≠ B @ A (generally)

# Transpose
A_T = A.T  # [[1, 3], [2, 4]]
# (A @ B)^T = B^T @ A^T

# Special matrices
I = np.eye(3)              # Identity matrix
zeros = np.zeros((3, 3))   # Zero matrix
ones = np.ones((3, 3))     # Ones matrix
diag = np.diag([1, 2, 3])  # Diagonal matrix

# Matrix properties
rank = np.linalg.matrix_rank(A)  # Number of linearly independent rows/cols
trace = np.trace(A)              # Sum of diagonal elements
det = np.linalg.det(A)           # Determinant

# Determinant interpretation:
# - det = 0: Matrix is singular (not invertible)
# - |det|: Scaling factor for volume transformation
# - sign(det): Orientation preservation/reversal

# Matrix inverse
A_inv = np.linalg.inv(A)
# A @ A_inv = I
# Only exists if det(A) ≠ 0

# Pseudo-inverse (Moore-Penrose)
A_pinv = np.linalg.pinv(A)
# Works for non-square and singular matrices
```

### Eigenvalues and Eigenvectors

```python
"""
Eigenvalue equation: Av = λv
- v: eigenvector (direction unchanged by transformation)
- λ: eigenvalue (scaling factor)

Geometric interpretation:
- Eigenvectors: directions that remain unchanged (only scaled)
- Eigenvalues: how much stretching/compression in that direction
"""

A = np.array([[4, 2], [1, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
# eigenvalues: [5, 2]
# eigenvectors: columns are the eigenvectors

# Verify: Av = λv
v = eigenvectors[:, 0]
lambda_val = eigenvalues[0]
print(np.allclose(A @ v, lambda_val * v))  # True

# Properties:
# - Sum of eigenvalues = trace(A)
# - Product of eigenvalues = det(A)
# - Symmetric matrices have real eigenvalues and orthogonal eigenvectors

# Eigendecomposition (spectral decomposition)
# A = V @ Λ @ V^(-1)
# Where V = eigenvector matrix, Λ = diagonal matrix of eigenvalues

V = eigenvectors
Lambda = np.diag(eigenvalues)
A_reconstructed = V @ Lambda @ np.linalg.inv(V)

# Applications in ML:
# 1. PCA: eigenvectors of covariance matrix are principal components
# 2. PageRank: principal eigenvector of link matrix
# 3. Spectral clustering: eigenvectors of Laplacian matrix
# 4. Stability analysis of dynamical systems
```

### Singular Value Decomposition (SVD)

```python
"""
SVD: A = U @ Σ @ V^T
- U: left singular vectors (m x m orthogonal matrix)
- Σ: singular values (m x n diagonal matrix)
- V^T: right singular vectors (n x n orthogonal matrix)

Works for ANY matrix (unlike eigendecomposition)
"""

A = np.array([[1, 2, 3], [4, 5, 6]])

U, S, Vt = np.linalg.svd(A, full_matrices=True)
# U: (2, 2) - orthonormal columns
# S: (2,) - singular values (always non-negative, sorted descending)
# Vt: (3, 3) - orthonormal rows

# Reconstruct A
Sigma = np.zeros((2, 3))
np.fill_diagonal(Sigma, S)
A_reconstructed = U @ Sigma @ Vt

# Reduced/Compact SVD (more efficient)
U_r, S_r, Vt_r = np.linalg.svd(A, full_matrices=False)
# U_r: (2, 2), Vt_r: (2, 3)

# Low-rank approximation (truncated SVD)
def truncated_svd(A, k):
    """Keep only top k singular values."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# Applications in ML:
# 1. Dimensionality reduction (similar to PCA)
# 2. Matrix completion (recommendation systems)
# 3. Image compression
# 4. Latent Semantic Analysis (NLP)
# 5. Noise reduction

# Relationship to eigendecomposition:
# A^T @ A has eigenvalues = S²  and eigenvectors = V
# A @ A^T has eigenvalues = S² and eigenvectors = U
```

### Important Matrix Decompositions

```python
# QR Decomposition: A = Q @ R
# Q: orthogonal matrix
# R: upper triangular matrix
Q, R = np.linalg.qr(A)
# Used in: solving linear systems, least squares

# Cholesky Decomposition: A = L @ L^T
# Only for symmetric positive definite matrices
# L: lower triangular matrix
L = np.linalg.cholesky(np.array([[4, 2], [2, 3]]))
# Used in: sampling from multivariate Gaussian, optimization

# LU Decomposition: A = P @ L @ U
# P: permutation matrix
# L: lower triangular
# U: upper triangular
from scipy.linalg import lu
P, L, U = lu(A)
# Used in: solving linear systems efficiently
```

---

## 2. Calculus

### Derivatives

```python
"""
Derivative: rate of change of a function

f'(x) = lim(h→0) [f(x+h) - f(x)] / h

Common derivatives:
- d/dx [x^n] = n * x^(n-1)
- d/dx [e^x] = e^x
- d/dx [ln(x)] = 1/x
- d/dx [sin(x)] = cos(x)
- d/dx [cos(x)] = -sin(x)
"""

import sympy as sp

x = sp.Symbol('x')

# Symbolic differentiation
f = x**3 + 2*x**2 + x
f_prime = sp.diff(f, x)  # 3x² + 4x + 1

# Numerical differentiation
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

# Chain rule: d/dx [f(g(x))] = f'(g(x)) * g'(x)
# Product rule: d/dx [f*g] = f'*g + f*g'
# Quotient rule: d/dx [f/g] = (f'*g - f*g') / g²
```

### Gradients and Partial Derivatives

```python
"""
Gradient: vector of partial derivatives
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

Points in direction of steepest ascent.
"""

# Example: f(x, y) = x² + y²
# ∂f/∂x = 2x
# ∂f/∂y = 2y
# ∇f = [2x, 2y]

def gradient_example(point):
    """Gradient of f(x,y) = x² + y²"""
    x, y = point
    return np.array([2*x, 2*y])

# Directional derivative
# Rate of change in direction u: ∇f · u
def directional_derivative(grad, direction):
    direction = direction / np.linalg.norm(direction)  # Normalize
    return np.dot(grad, direction)

# Jacobian matrix (for vector-valued functions)
# f: R^n → R^m
# J[i,j] = ∂f_i/∂x_j
# Shape: (m, n)

def jacobian_example():
    """
    f(x, y) = [x²y, 5x + sin(y)]
    J = [[2xy, x²], [5, cos(y)]]
    """
    pass

# Hessian matrix (second derivatives)
# H[i,j] = ∂²f/∂x_i∂x_j
# Used for: optimization (Newton's method), curvature analysis

def hessian_example():
    """
    f(x, y) = x² + y²
    H = [[2, 0], [0, 2]]
    """
    pass
```

### Chain Rule in Neural Networks (Backpropagation)

```python
"""
Backpropagation uses chain rule to compute gradients efficiently.

Forward pass: compute outputs
Backward pass: compute gradients from output to input

Example: y = relu(W₂ @ relu(W₁ @ x + b₁) + b₂)
"""

import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Simple 2-layer network backward pass
def backward_pass(x, y_true, W1, b1, W2, b2):
    # Forward pass
    z1 = W1 @ x + b1
    a1 = relu(z1)
    z2 = W2 @ a1 + b2
    y_pred = sigmoid(z2)
    
    # Backward pass (chain rule)
    # Loss = BCE = -[y*log(p) + (1-y)*log(1-p)]
    # dL/dp = -y/p + (1-y)/(1-p)
    
    m = x.shape[1]  # batch size
    
    # Output layer
    dz2 = y_pred - y_true  # Combined gradient for sigmoid + BCE
    dW2 = (1/m) * dz2 @ a1.T
    db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)
    
    # Hidden layer
    da1 = W2.T @ dz2
    dz1 = da1 * relu_derivative(z1)
    dW1 = (1/m) * dz1 @ x.T
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2
```

### Taylor Series

```python
"""
Taylor series: approximate function with polynomials

f(x) ≈ f(a) + f'(a)(x-a) + f''(a)(x-a)²/2! + ...

Used in:
- Function approximation
- Optimization (second-order methods)
- Understanding activation functions
"""

# First-order Taylor (linear approximation)
# f(x + Δx) ≈ f(x) + f'(x) * Δx

# Second-order Taylor (quadratic approximation)
# f(x + Δx) ≈ f(x) + f'(x) * Δx + 0.5 * f''(x) * Δx²

# Multivariable Taylor:
# f(x + Δx) ≈ f(x) + ∇f(x)ᵀΔx + 0.5 * ΔxᵀH(x)Δx
```

---

## 3. Probability Theory

### Basic Probability

```python
"""
Probability axioms:
1. P(A) ≥ 0 for all events A
2. P(Ω) = 1 (sample space)
3. P(A ∪ B) = P(A) + P(B) if A and B are mutually exclusive

Key formulas:
- P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
- P(A | B) = P(A ∩ B) / P(B)  (conditional probability)
- P(A ∩ B) = P(A | B) * P(B) = P(B | A) * P(A)
"""

# Independence: P(A ∩ B) = P(A) * P(B)
# If A and B independent, P(A | B) = P(A)

# Conditional independence: A ⊥ B | C
# P(A, B | C) = P(A | C) * P(B | C)
```

### Bayes' Theorem

```python
"""
Bayes' Theorem:
P(A | B) = P(B | A) * P(A) / P(B)

Components:
- P(A | B): Posterior (what we want)
- P(B | A): Likelihood (data given hypothesis)
- P(A): Prior (initial belief)
- P(B): Evidence (normalizing constant)
"""

# Example: Medical diagnosis
# Disease prevalence: P(D) = 0.01
# Test sensitivity: P(+|D) = 0.95
# Test specificity: P(-|¬D) = 0.90

def bayes_medical_diagnosis():
    P_D = 0.01               # Prior: P(disease)
    P_pos_given_D = 0.95     # Sensitivity: P(+|D)
    P_neg_given_notD = 0.90  # Specificity: P(-|¬D)
    P_pos_given_notD = 1 - P_neg_given_notD  # False positive rate
    
    # P(+) = P(+|D)P(D) + P(+|¬D)P(¬D)
    P_pos = P_pos_given_D * P_D + P_pos_given_notD * (1 - P_D)
    
    # P(D|+) = P(+|D) * P(D) / P(+)
    P_D_given_pos = (P_pos_given_D * P_D) / P_pos
    
    return P_D_given_pos  # ≈ 0.087 (only 8.7%!)

# Application in ML:
# Naive Bayes classifier
# P(class | features) ∝ P(features | class) * P(class)
```

### Probability Distributions

```python
import numpy as np
from scipy import stats

# Discrete Distributions

# Bernoulli: single binary trial
# P(X=1) = p, P(X=0) = 1-p
bernoulli = stats.bernoulli(p=0.3)
print(f"PMF: P(X=1) = {bernoulli.pmf(1)}")

# Binomial: n Bernoulli trials
# P(X=k) = C(n,k) * p^k * (1-p)^(n-k)
binomial = stats.binom(n=10, p=0.3)
print(f"Mean: {binomial.mean()}, Var: {binomial.var()}")

# Poisson: rare events in fixed interval
# P(X=k) = (λ^k * e^(-λ)) / k!
poisson = stats.poisson(mu=5)

# Categorical/Multinomial: multiple categories
# Generalization of Bernoulli/Binomial

# Continuous Distributions

# Uniform: all values equally likely
uniform = stats.uniform(loc=0, scale=1)  # [0, 1]

# Normal/Gaussian
# f(x) = (1/√(2πσ²)) * exp(-(x-μ)²/(2σ²))
normal = stats.norm(loc=0, scale=1)  # Standard normal
print(f"PDF at 0: {normal.pdf(0)}")
print(f"CDF at 1.96: {normal.cdf(1.96)}")  # ≈ 0.975

# Multivariate Normal
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
mvn = stats.multivariate_normal(mean=mean, cov=cov)

# Exponential: time between Poisson events
# f(x) = λ * exp(-λx)
exponential = stats.expon(scale=1/5)  # λ = 5

# Gamma: sum of exponential distributions
gamma = stats.gamma(a=2, scale=1)

# Beta: probability of probability
# Conjugate prior for Bernoulli/Binomial
beta = stats.beta(a=2, b=5)

# Key properties
dist = stats.norm(0, 1)
mean = dist.mean()
var = dist.var()
std = dist.std()
samples = dist.rvs(size=1000)  # Random samples
percentile = dist.ppf(0.95)    # 95th percentile
```

### Expectation and Variance

```python
"""
Expected Value (Mean):
E[X] = Σ x * P(X=x)           (discrete)
E[X] = ∫ x * f(x) dx          (continuous)

Properties:
- E[aX + b] = a*E[X] + b
- E[X + Y] = E[X] + E[Y]      (always)
- E[XY] = E[X]*E[Y]           (if independent)

Variance:
Var(X) = E[(X - μ)²] = E[X²] - E[X]²

Properties:
- Var(aX + b) = a² * Var(X)
- Var(X + Y) = Var(X) + Var(Y)  (if independent)

Covariance:
Cov(X, Y) = E[(X - μ_X)(Y - μ_Y)] = E[XY] - E[X]E[Y]

Correlation:
ρ(X, Y) = Cov(X, Y) / (σ_X * σ_Y)
-1 ≤ ρ ≤ 1
"""

# Compute from data
data = np.random.normal(0, 1, 1000)
mean = np.mean(data)
var = np.var(data)
std = np.std(data)

# Covariance matrix
X = np.random.randn(100, 3)
cov_matrix = np.cov(X.T)

# Correlation matrix
corr_matrix = np.corrcoef(X.T)
```

### Information Theory

```python
"""
Entropy: measure of uncertainty
H(X) = -Σ P(x) * log₂(P(x))

Cross-entropy:
H(P, Q) = -Σ P(x) * log(Q(x))
Measures "surprise" of using Q to encode data from P

KL Divergence:
D_KL(P || Q) = Σ P(x) * log(P(x)/Q(x)) = H(P, Q) - H(P)
Measures difference between distributions
NOT symmetric: D_KL(P||Q) ≠ D_KL(Q||P)
"""

def entropy(probs):
    """Shannon entropy."""
    probs = np.array(probs)
    probs = probs[probs > 0]  # Avoid log(0)
    return -np.sum(probs * np.log2(probs))

def cross_entropy(p, q):
    """Cross entropy H(P, Q)."""
    p, q = np.array(p), np.array(q)
    return -np.sum(p * np.log(q + 1e-10))

def kl_divergence(p, q):
    """KL divergence D_KL(P || Q)."""
    p, q = np.array(p), np.array(q)
    return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))

# Binary cross-entropy (classification loss)
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-10) + 
                    (1 - y_true) * np.log(1 - y_pred + 1e-10))

# Categorical cross-entropy (multi-class)
def categorical_cross_entropy(y_true, y_pred):
    # y_true: one-hot encoded
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-10), axis=1))
```

---

## 4. Statistics

### Descriptive Statistics

```python
import numpy as np
from scipy import stats

data = np.random.randn(1000)

# Central tendency
mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data, keepdims=True).mode[0]

# Dispersion
variance = np.var(data)
std_dev = np.std(data)
range_val = np.max(data) - np.min(data)
iqr = np.percentile(data, 75) - np.percentile(data, 25)

# Shape
skewness = stats.skew(data)   # Asymmetry (0 = symmetric)
kurtosis = stats.kurtosis(data)  # Tail heaviness (0 = normal)

# Percentiles/Quantiles
percentiles = np.percentile(data, [25, 50, 75])
```

### Statistical Inference

```python
"""
Parameter estimation:
1. Point estimation: single value (mean, MLE)
2. Interval estimation: confidence intervals

Sampling distribution:
- Distribution of a statistic over many samples
- Standard error = std / sqrt(n)
"""

# Confidence Intervals
def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)  # Standard error
    
    # t-distribution for small samples
    ci = stats.t.interval(confidence, df=n-1, loc=mean, scale=se)
    return ci

# Central Limit Theorem
# Sample means approach normal distribution as n → ∞
# Mean of sample means = population mean
# Std of sample means = population std / sqrt(n)
```

### Hypothesis Testing

```python
"""
Steps:
1. State null (H₀) and alternative (H₁) hypotheses
2. Choose significance level α (typically 0.05)
3. Calculate test statistic
4. Compute p-value
5. Reject H₀ if p-value < α

Errors:
- Type I (α): Reject H₀ when it's true (false positive)
- Type II (β): Fail to reject H₀ when it's false (false negative)
- Power = 1 - β: Probability of correctly rejecting H₀
"""

# One-sample t-test
# H₀: μ = μ₀
data = np.random.normal(5, 1, 100)
t_stat, p_value = stats.ttest_1samp(data, popmean=5)
print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")

# Two-sample t-test
# H₀: μ₁ = μ₂
group1 = np.random.normal(5, 1, 100)
group2 = np.random.normal(5.5, 1, 100)
t_stat, p_value = stats.ttest_ind(group1, group2)

# Paired t-test
# H₀: mean difference = 0
before = np.random.normal(100, 10, 50)
after = before + np.random.normal(5, 3, 50)
t_stat, p_value = stats.ttest_rel(before, after)

# Chi-square test (categorical data)
# H₀: observed = expected
observed = [50, 30, 20]
expected = [40, 35, 25]
chi2, p_value = stats.chisquare(observed, expected)

# Chi-square test of independence
contingency_table = np.array([[10, 20], [15, 25]])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

# ANOVA (comparing multiple groups)
# H₀: μ₁ = μ₂ = ... = μₖ
group1 = np.random.normal(5, 1, 50)
group2 = np.random.normal(5.5, 1, 50)
group3 = np.random.normal(6, 1, 50)
f_stat, p_value = stats.f_oneway(group1, group2, group3)

# Multiple testing correction
from statsmodels.stats.multitest import multipletests
p_values = [0.01, 0.04, 0.03, 0.001]
rejected, corrected_p, _, _ = multipletests(p_values, method='bonferroni')
```

### Maximum Likelihood Estimation (MLE)

```python
"""
MLE: Find parameters θ that maximize P(data | θ)

L(θ) = P(X | θ) = Π P(xᵢ | θ)
log L(θ) = Σ log P(xᵢ | θ)  (log-likelihood, easier to optimize)

Solution: ∂log L / ∂θ = 0
"""

import numpy as np
from scipy.optimize import minimize

# MLE for Normal distribution
def normal_log_likelihood(params, data):
    mu, sigma = params
    n = len(data)
    log_likelihood = -n/2 * np.log(2*np.pi*sigma**2) - \
                     np.sum((data - mu)**2) / (2*sigma**2)
    return -log_likelihood  # Minimize negative log-likelihood

data = np.random.normal(5, 2, 1000)
result = minimize(normal_log_likelihood, x0=[0, 1], args=(data,))
print(f"MLE estimates: μ={result.x[0]:.2f}, σ={result.x[1]:.2f}")

# Closed-form MLE for Normal:
# μ_MLE = sample mean
# σ²_MLE = sample variance (biased, with n not n-1)


# Maximum A Posteriori (MAP)
# θ_MAP = argmax P(θ | X) = argmax [P(X | θ) * P(θ)]
# Includes prior distribution P(θ)
# With uninformative prior: MAP = MLE
```

---

## 5. Optimization

### Gradient Descent

```python
"""
Gradient Descent: iterative optimization
θ_new = θ_old - α * ∇f(θ)

Variants:
- Batch GD: Use all data
- Stochastic GD (SGD): Use one sample
- Mini-batch GD: Use subset of data
"""

def gradient_descent(f, grad_f, x0, learning_rate=0.01, n_iters=1000, tol=1e-6):
    """Basic gradient descent."""
    x = x0.copy()
    history = [x.copy()]
    
    for _ in range(n_iters):
        gradient = grad_f(x)
        x_new = x - learning_rate * gradient
        
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
        history.append(x.copy())
    
    return x, history

# Example: minimize f(x, y) = x² + y²
def f(x):
    return x[0]**2 + x[1]**2

def grad_f(x):
    return np.array([2*x[0], 2*x[1]])

x_opt, history = gradient_descent(f, grad_f, np.array([5.0, 5.0]))
```

### Advanced Optimizers

```python
"""
Momentum: accumulate gradients for faster convergence
v = β*v + (1-β)*∇f
θ = θ - α*v

RMSprop: adaptive learning rate per parameter
s = β*s + (1-β)*(∇f)²
θ = θ - α * ∇f / sqrt(s + ε)

Adam: combines momentum + adaptive learning rates
m = β₁*m + (1-β₁)*∇f        (first moment)
v = β₂*v + (1-β₂)*(∇f)²     (second moment)
m̂ = m / (1-β₁^t)            (bias correction)
v̂ = v / (1-β₂^t)
θ = θ - α * m̂ / sqrt(v̂ + ε)
"""

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0
    
    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        
        self.t += 1
        
        # Update biased moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        # Update parameters
        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params
```

### Convexity and Optimization

```python
"""
Convex function: f(λx + (1-λ)y) ≤ λf(x) + (1-λ)f(y)
- Bowl-shaped
- Local minimum = global minimum
- Gradient descent guaranteed to find optimum

Checking convexity:
- Hessian matrix is positive semi-definite (all eigenvalues ≥ 0)

Non-convex challenges (neural networks):
- Multiple local minima
- Saddle points
- Plateaus
"""

def is_convex(hessian):
    """Check if function is convex via Hessian."""
    eigenvalues = np.linalg.eigvals(hessian)
    return np.all(eigenvalues >= 0)

# Regularization creates convexity
# L2 regularization: adds λ||w||² (strongly convex if λ > 0)
```

---

## 6. Interview Questions

```python
# Q1: Why do we use log-likelihood instead of likelihood in MLE?
"""
1. Numerical stability: Products of small probabilities → underflow
2. Converts products to sums: easier to differentiate
3. Monotonic: same optimum as likelihood
4. Computational efficiency: sums are faster than products
"""


# Q2: Explain the difference between eigenvalue decomposition and SVD
"""
Eigendecomposition:
- Only for square matrices
- A = VΛV⁻¹
- Eigenvectors may not be orthogonal

SVD:
- Works for any matrix (m x n)
- A = UΣVᵀ
- U and V are always orthogonal
- Singular values are always real and non-negative
"""


# Q3: What is the relationship between covariance and correlation?
"""
Correlation = Cov(X,Y) / (σ_X * σ_Y)
- Covariance: unbounded, scale-dependent
- Correlation: bounded [-1, 1], scale-invariant
- Correlation = 0 doesn't imply independence (only linear)
"""


# Q4: Explain p-value in simple terms
"""
p-value: Probability of observing data this extreme or more,
assuming the null hypothesis is true.

Small p-value (<0.05) → unlikely under H₀ → reject H₀
Large p-value → data consistent with H₀ → fail to reject H₀

Common mistakes:
- p-value is NOT probability that H₀ is true
- Not rejecting ≠ accepting H₀
"""


# Q5: Why is cross-entropy used as loss for classification?
"""
1. Derived from MLE: maximizing likelihood = minimizing cross-entropy
2. Information theoretic: measures surprise/uncertainty
3. Gradient behavior: stronger gradients when wrong (vs MSE)
4. Proper scoring rule: optimal when predicting true probabilities
"""


# Q6: What's the difference between gradient descent variants?
"""
Batch GD: Use all data → stable but slow
SGD: Use 1 sample → fast but noisy
Mini-batch: Use subset → balance of speed and stability

For large datasets: mini-batch preferred
For small datasets: batch GD is fine
"""


# Q7: Explain bias-variance tradeoff mathematically
"""
E[(y - f̂(x))²] = Bias(f̂)² + Var(f̂) + σ²

Bias: E[f̂(x)] - f(x) → underfitting
Variance: E[(f̂(x) - E[f̂(x)])²] → overfitting
σ²: irreducible error (noise)

Cannot minimize both simultaneously → tradeoff
"""


# Q8: When would you use L1 vs L2 regularization?
"""
L1 (Lasso): |w|
- Sparse solutions (feature selection)
- Robust to outliers
- Non-differentiable at 0

L2 (Ridge): w²
- Smooth solutions
- Handles correlated features better
- Differentiable everywhere
- Closed-form solution exists

ElasticNet: combines both
"""


# Q9: Explain the Central Limit Theorem and its importance
"""
CLT: Sample means approach normal distribution as n → ∞,
regardless of underlying distribution (if finite variance).

Importance:
1. Justifies use of normal distribution in many tests
2. Enables confidence intervals for means
3. Foundation of many statistical methods
4. Works for sample sizes n ≥ 30 typically
"""


# Q10: What is the curse of dimensionality?
"""
In high dimensions:
1. Data becomes sparse: volume grows exponentially
2. Distances become similar: all points roughly equidistant
3. More data needed: exponentially more samples required
4. Overfitting risk: more features = more parameters

Solutions:
- Dimensionality reduction (PCA, t-SNE)
- Feature selection
- Regularization
- Careful feature engineering
"""
```
